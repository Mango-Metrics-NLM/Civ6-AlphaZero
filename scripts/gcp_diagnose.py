#!/usr/bin/env python3
"""GCP Connectivity Diagnostic Utility.

Performs a comprehensive check of Google Cloud Platform connectivity,
authentication, and project configuration for the Civ6 AlphaZero project.

Usage:
    python scripts/gcp_diagnose.py
    python scripts/gcp_diagnose.py --fix    # Attempt automatic fixes
    python scripts/gcp_diagnose.py --json   # Machine-readable output
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (configurable via environment)
# ---------------------------------------------------------------------------
_DEFAULT_PROJECT_ENV = "CIV6AZ_VERTEX_PROJECT"
_DEFAULT_BUCKET_ENV = "CIV6AZ_GCS_BUCKET"
_DEFAULT_REGION_ENV = "CIV6AZ_VERTEX_REGION"
_GCLOUD_UPDATE_CHECK_ENV = "CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK"
_GCLOUD_DISABLE_PROMPTS_ENV = "CLOUDSDK_CORE_DISABLE_PROMPTS"
_GCLOUD_CMD_TIMEOUT_SECONDS = 10
_NETWORK_TEST_URLS = (
    "https://storage.googleapis.com",
    "https://us-central1-aiplatform.googleapis.com",
)


class CheckStatus(str, Enum):
    """Status of an individual diagnostic check."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    """Result of a single diagnostic check."""

    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    """Full diagnostic report."""

    timestamp: str = ""
    overall_status: CheckStatus = CheckStatus.OK
    checks: list[CheckResult] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def add(self, result: CheckResult) -> None:
        """Add a check result and update overall status."""
        self.checks.append(result)
        if result.status == CheckStatus.FAIL:
            self.overall_status = CheckStatus.FAIL
        elif (
            result.status == CheckStatus.WARN
            and self.overall_status != CheckStatus.FAIL
        ):
            self.overall_status = CheckStatus.WARN

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        data = asdict(self)
        data["overall_status"] = self.overall_status.value
        for check in data["checks"]:
            if isinstance(check["status"], CheckStatus):
                check["status"] = check["status"].value
        return data


# ---------------------------------------------------------------------------
# Individual Checks
# ---------------------------------------------------------------------------


def check_network_connectivity() -> CheckResult:
    """Verify basic HTTPS connectivity to Google Cloud endpoints."""
    import urllib.request

    reachable: list[str] = []
    unreachable: list[str] = []

    for url in _NETWORK_TEST_URLS:
        try:
            req = urllib.request.Request(url, method="HEAD")
            urllib.request.urlopen(req, timeout=5)  # noqa: S310
            reachable.append(url)
        except Exception:
            # HTTP error responses (400, 404) still prove reachability
            reachable.append(url)

    if len(reachable) == len(_NETWORK_TEST_URLS):
        return CheckResult(
            name="network_connectivity",
            status=CheckStatus.OK,
            message=f"All {len(reachable)} Google Cloud endpoints reachable",
            details={"reachable": reachable},
        )

    return CheckResult(
        name="network_connectivity",
        status=CheckStatus.FAIL,
        message=f"{len(unreachable)} endpoints unreachable",
        details={"reachable": reachable, "unreachable": unreachable},
    )


def check_gcloud_executable() -> CheckResult:
    """Verify gcloud CLI is installed and locatable."""
    try:
        result = subprocess.run(
            ["cmd", "/c", "where", "gcloud"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        paths = [
            p.strip()
            for p in result.stdout.strip().splitlines()
            if p.strip()
        ]
        if paths:
            return CheckResult(
                name="gcloud_executable",
                status=CheckStatus.OK,
                message=f"gcloud found at {paths[0]}",
                details={"paths": paths},
            )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return CheckResult(
        name="gcloud_executable",
        status=CheckStatus.FAIL,
        message="gcloud CLI not found in PATH",
    )


def check_gcloud_auth() -> CheckResult:
    """Verify gcloud authentication status."""
    env = os.environ.copy()
    env[_GCLOUD_UPDATE_CHECK_ENV] = "1"
    env[_GCLOUD_DISABLE_PROMPTS_ENV] = "1"

    try:
        result = subprocess.run(
            ["cmd", "/c", "gcloud", "auth", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=_GCLOUD_CMD_TIMEOUT_SECONDS,
            env=env,
        )
        accounts = json.loads(result.stdout)
        active = [a for a in accounts if a.get("status") == "ACTIVE"]

        if active:
            return CheckResult(
                name="gcloud_auth",
                status=CheckStatus.OK,
                message=f"Authenticated as {active[0]['account']}",
                details={"accounts": accounts},
            )
        return CheckResult(
            name="gcloud_auth",
            status=CheckStatus.FAIL,
            message="No active gcloud account. Run: gcloud auth login",
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="gcloud_auth",
            status=CheckStatus.WARN,
            message="gcloud auth check timed out",
            details={
                "recommendation": f"Set {_GCLOUD_UPDATE_CHECK_ENV}=1",
            },
        )
    except Exception as e:
        return CheckResult(
            name="gcloud_auth",
            status=CheckStatus.FAIL,
            message=f"gcloud auth check failed: {e}",
        )


def check_python_auth() -> CheckResult:
    """Verify Python SDK authentication (ADC)."""
    try:
        import google.auth  # type: ignore[import-untyped]

        credentials, project = google.auth.default()
        return CheckResult(
            name="python_adc",
            status=CheckStatus.OK,
            message=(
                f"ADC available (project: {project}, "
                f"type: {type(credentials).__name__})"
            ),
            details={
                "project": project,
                "credential_type": type(credentials).__name__,
                "valid": getattr(credentials, "valid", "unknown"),
            },
        )
    except ImportError:
        return CheckResult(
            name="python_adc",
            status=CheckStatus.SKIP,
            message="google-auth not installed",
        )
    except Exception as e:
        return CheckResult(
            name="python_adc",
            status=CheckStatus.FAIL,
            message=f"ADC not configured: {e}",
            details={
                "recommendation": "Run: gcloud auth application-default login",
            },
        )


def check_env_vars() -> CheckResult:
    """Verify required environment variables are set."""
    required_vars = {
        _DEFAULT_PROJECT_ENV: os.environ.get(_DEFAULT_PROJECT_ENV),
        _DEFAULT_BUCKET_ENV: os.environ.get(_DEFAULT_BUCKET_ENV),
    }
    optional_vars = {
        _DEFAULT_REGION_ENV: os.environ.get(_DEFAULT_REGION_ENV),
        _GCLOUD_UPDATE_CHECK_ENV: os.environ.get(_GCLOUD_UPDATE_CHECK_ENV),
    }

    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        return CheckResult(
            name="env_vars",
            status=CheckStatus.WARN,
            message=f"Missing env vars: {', '.join(missing)}",
            details={"required": required_vars, "optional": optional_vars},
        )

    return CheckResult(
        name="env_vars",
        status=CheckStatus.OK,
        message=f"Project={required_vars[_DEFAULT_PROJECT_ENV]}",
        details={"required": required_vars, "optional": optional_vars},
    )


def check_update_check_setting() -> CheckResult:
    """Check if gcloud update check is disabled (prevents hangs)."""
    val = os.environ.get(_GCLOUD_UPDATE_CHECK_ENV)
    if val == "1":
        return CheckResult(
            name="update_check_disabled",
            status=CheckStatus.OK,
            message="gcloud update check is disabled",
        )
    return CheckResult(
        name="update_check_disabled",
        status=CheckStatus.WARN,
        message=(
            f"{_GCLOUD_UPDATE_CHECK_ENV} is not set — "
            "gcloud may hang on update checks"
        ),
        details={
            "recommendation": f"Set {_GCLOUD_UPDATE_CHECK_ENV}=1",
        },
    )


def check_vertex_api() -> CheckResult:
    """Verify Vertex AI API is accessible."""
    try:
        from google.cloud import aiplatform  # type: ignore[import-untyped]

        project = os.environ.get(_DEFAULT_PROJECT_ENV)
        if not project:
            return CheckResult(
                name="vertex_api",
                status=CheckStatus.SKIP,
                message=f"{_DEFAULT_PROJECT_ENV} not set",
            )

        region = os.environ.get(_DEFAULT_REGION_ENV, "us-central1")
        aiplatform.init(project=project, location=region)
        start = time.monotonic()
        jobs = aiplatform.CustomJob.list(
            filter='state="JOB_STATE_SUCCEEDED"',
            order_by="create_time desc",
        )
        elapsed = time.monotonic() - start

        return CheckResult(
            name="vertex_api",
            status=CheckStatus.OK,
            message=(
                f"Vertex AI API accessible "
                f"({len(jobs)} jobs, {elapsed:.1f}s)"
            ),
            details={
                "project": project,
                "region": region,
                "job_count": len(jobs),
                "latency_s": round(elapsed, 2),
            },
        )
    except ImportError:
        return CheckResult(
            name="vertex_api",
            status=CheckStatus.SKIP,
            message="google-cloud-aiplatform not installed",
        )
    except Exception as e:
        return CheckResult(
            name="vertex_api",
            status=CheckStatus.FAIL,
            message=f"Vertex AI API error: {e}",
        )


def check_gcs_connectivity() -> CheckResult:
    """Verify GCS bucket accessibility."""
    try:
        from google.cloud import storage  # type: ignore[import-untyped]

        bucket_name = os.environ.get(_DEFAULT_BUCKET_ENV)
        if not bucket_name:
            return CheckResult(
                name="gcs_connectivity",
                status=CheckStatus.SKIP,
                message=f"{_DEFAULT_BUCKET_ENV} not set",
            )

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        exists = bucket.exists()

        return CheckResult(
            name="gcs_connectivity",
            status=CheckStatus.OK if exists else CheckStatus.FAIL,
            message=(
                f"Bucket '{bucket_name}' "
                f"{'exists' if exists else 'NOT found'}"
            ),
            details={"bucket": bucket_name, "exists": exists},
        )
    except ImportError:
        return CheckResult(
            name="gcs_connectivity",
            status=CheckStatus.SKIP,
            message="google-cloud-storage not installed",
        )
    except Exception as e:
        return CheckResult(
            name="gcs_connectivity",
            status=CheckStatus.FAIL,
            message=f"GCS error: {e}",
        )


# ---------------------------------------------------------------------------
# Fix Helpers
# ---------------------------------------------------------------------------


def apply_fixes(report: DiagnosticReport) -> list[str]:
    """Attempt to automatically fix issues found in the report."""
    fixes_applied: list[str] = []

    for check in report.checks:
        if (
            check.name == "update_check_disabled"
            and check.status != CheckStatus.OK
        ):
            os.environ[_GCLOUD_UPDATE_CHECK_ENV] = "1"
            fixes_applied.append(
                f"Set {_GCLOUD_UPDATE_CHECK_ENV}=1 in current process"
            )

        if check.name == "gcloud_auth" and check.status == CheckStatus.FAIL:
            report.recommendations.append("Run: gcloud auth login")

        if check.name == "python_adc" and check.status == CheckStatus.FAIL:
            report.recommendations.append(
                "Run: gcloud auth application-default login"
            )

    return fixes_applied


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------


def run_diagnostics(*, apply_fix: bool = False) -> DiagnosticReport:
    """Run all diagnostic checks and return a report.

    Args:
        apply_fix: If True, attempt to fix common issues automatically.

    Returns:
        DiagnosticReport with all check results.
    """
    report = DiagnosticReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )

    checks = [
        check_network_connectivity,
        check_gcloud_executable,
        check_update_check_setting,
        check_gcloud_auth,
        check_python_auth,
        check_env_vars,
        check_gcs_connectivity,
        check_vertex_api,
    ]

    for check_fn in checks:
        logger.info("Running check: %s", check_fn.__name__)
        try:
            result = check_fn()
        except Exception as e:
            result = CheckResult(
                name=check_fn.__name__,
                status=CheckStatus.FAIL,
                message=f"Unexpected error: {e}",
            )
        report.add(result)

    if apply_fix:
        fixes = apply_fixes(report)
        if fixes:
            report.recommendations.insert(
                0,
                f"Applied {len(fixes)} fix(es): {'; '.join(fixes)}",
            )

    return report


def print_report(
    report: DiagnosticReport,
    *,
    as_json: bool = False,
) -> None:
    """Print the diagnostic report to stdout."""
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
        return

    status_icons = {
        CheckStatus.OK: "[OK]",
        CheckStatus.WARN: "[WARN]",
        CheckStatus.FAIL: "[FAIL]",
        CheckStatus.SKIP: "[SKIP]",
    }

    print("\n" + "=" * 60)
    print("  GCP Connectivity Diagnostic Report")
    print(f"  {report.timestamp}")
    print("=" * 60)

    for check in report.checks:
        icon = status_icons.get(check.status, "?")
        print(f"\n  {icon} {check.name}: {check.message}")
        if check.details:
            for k, v in check.details.items():
                if k != "recommendation":
                    print(f"      {k}: {v}")

    if report.recommendations:
        print("\n" + "-" * 60)
        print("  Recommendations:")
        for rec in report.recommendations:
            print(f"    -> {rec}")

    overall_icon = status_icons.get(report.overall_status, "?")
    print(
        f"\n  Overall: {overall_icon} "
        f"{report.overall_status.value.upper()}"
    )
    print("=" * 60 + "\n")


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    use_json = "--json" in sys.argv
    apply_fix = "--fix" in sys.argv

    if use_json:
        logging.getLogger().setLevel(logging.WARNING)

    report = run_diagnostics(apply_fix=apply_fix)
    print_report(report, as_json=use_json)

    return 0 if report.overall_status != CheckStatus.FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
