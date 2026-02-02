#!/usr/bin/env python3
"""GCP credential and environment validation utility.

Validates GCP credentials and configuration for E2E testing.
Provides detailed diagnostics for troubleshooting authentication issues.

Usage:
    python scripts/validate_gcp_credentials.py

Environment Variables:
    GCP_PROJECT_ID: GCP project identifier
    GCS_BUCKET_NAME: GCS bucket name for checkpoints
    GOOGLE_APPLICATION_CREDENTIALS: Path to service account key (optional)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Status of a validation check."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    name: str
    status: ValidationStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status_emoji = {
            ValidationStatus.PASS: "✅",
            ValidationStatus.FAIL: "❌",
            ValidationStatus.WARN: "⚠️",
            ValidationStatus.SKIP: "⏭️",
        }
        return f"{status_emoji[self.status]} {self.name}: {self.message}"


@dataclass
class ValidationReport:
    """Complete validation report."""

    results: list[ValidationResult] = field(default_factory=list)

    def add(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)

    @property
    def passed(self) -> bool:
        """Check if all required validations passed."""
        return all(
            r.status in (ValidationStatus.PASS, ValidationStatus.WARN, ValidationStatus.SKIP)
            for r in self.results
        )

    @property
    def summary(self) -> dict[str, int]:
        """Get summary counts by status."""
        counts: dict[str, int] = {s.value: 0 for s in ValidationStatus}
        for r in self.results:
            counts[r.status.value] += 1
        return counts

    def print_report(self) -> None:
        """Print formatted report."""
        logger.info("\n" + "=" * 60)
        logger.info("GCP CREDENTIAL VALIDATION REPORT")
        logger.info("=" * 60)

        for result in self.results:
            logger.info(str(result))
            if result.details:
                for key, value in result.details.items():
                    logger.info(f"    {key}: {value}")

        logger.info("-" * 60)
        summary = self.summary
        logger.info(
            f"Summary: {summary['PASS']} passed, {summary['FAIL']} failed, "
            f"{summary['WARN']} warnings, {summary['SKIP']} skipped"
        )
        logger.info("=" * 60)


class GCPCredentialValidator:
    """Validates GCP credentials and configuration."""

    def __init__(self) -> None:
        """Initialize validator."""
        self.report = ValidationReport()

    def validate_all(self) -> ValidationReport:
        """Run all validation checks."""
        self._check_environment_variables()
        self._check_credentials_file()
        self._check_application_default_credentials()
        self._check_gcs_access()
        self._check_vertex_ai_access()
        self._check_cloud_logging_access()
        return self.report

    def _check_environment_variables(self) -> None:
        """Check required environment variables."""
        required_vars = {
            "GCP_PROJECT_ID": os.environ.get("GCP_PROJECT_ID", ""),
            "GCS_BUCKET_NAME": os.environ.get("GCS_BUCKET_NAME", ""),
        }

        optional_vars = {
            "GCP_REGION": os.environ.get("GCP_REGION", "us-central1"),
            "GOOGLE_APPLICATION_CREDENTIALS": os.environ.get(
                "GOOGLE_APPLICATION_CREDENTIALS", ""
            ),
        }

        # Check required variables
        missing = [k for k, v in required_vars.items() if not v]
        if missing:
            self.report.add(ValidationResult(
                name="Environment Variables",
                status=ValidationStatus.FAIL,
                message=f"Missing required: {', '.join(missing)}",
                details={"required": required_vars, "optional": optional_vars},
            ))
        else:
            self.report.add(ValidationResult(
                name="Environment Variables",
                status=ValidationStatus.PASS,
                message="All required variables set",
                details={"project_id": required_vars["GCP_PROJECT_ID"][:20] + "..."},
            ))

    def _check_credentials_file(self) -> None:
        """Check for explicit credentials file."""
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

        if not creds_path:
            self.report.add(ValidationResult(
                name="Service Account Key",
                status=ValidationStatus.SKIP,
                message="GOOGLE_APPLICATION_CREDENTIALS not set (using ADC)",
            ))
            return

        path = Path(creds_path)
        if not path.exists():
            self.report.add(ValidationResult(
                name="Service Account Key",
                status=ValidationStatus.FAIL,
                message=f"Credentials file not found: {path}",
            ))
            return

        # Validate JSON structure
        try:
            with open(path, encoding="utf-8") as f:
                creds = json.load(f)

            cred_type = creds.get("type", "unknown")
            if cred_type == "service_account":
                self.report.add(ValidationResult(
                    name="Service Account Key",
                    status=ValidationStatus.PASS,
                    message="Valid service account credentials",
                    details={"email": creds.get("client_email", "N/A")},
                ))
            else:
                self.report.add(ValidationResult(
                    name="Service Account Key",
                    status=ValidationStatus.WARN,
                    message=f"Unexpected credential type: {cred_type}",
                ))
        except json.JSONDecodeError as e:
            self.report.add(ValidationResult(
                name="Service Account Key",
                status=ValidationStatus.FAIL,
                message=f"Invalid JSON in credentials file: {e}",
            ))

    def _check_application_default_credentials(self) -> None:
        """Check for application default credentials."""
        # Determine ADC path based on OS
        if os.name == "nt":  # Windows
            adc_path = Path(os.environ.get("APPDATA", "")) / "gcloud" / "application_default_credentials.json"
        else:  # Linux/Mac
            adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"

        if adc_path.exists():
            try:
                with open(adc_path, encoding="utf-8") as f:
                    adc = json.load(f)

                adc_type = adc.get("type", "unknown")
                self.report.add(ValidationResult(
                    name="Application Default Credentials",
                    status=ValidationStatus.PASS,
                    message=f"ADC found ({adc_type})",
                    details={"path": str(adc_path)},
                ))
            except (json.JSONDecodeError, OSError) as e:
                self.report.add(ValidationResult(
                    name="Application Default Credentials",
                    status=ValidationStatus.FAIL,
                    message=f"Error reading ADC: {e}",
                ))
        else:
            self.report.add(ValidationResult(
                name="Application Default Credentials",
                status=ValidationStatus.WARN,
                message="ADC not found. Run: gcloud auth application-default login",
                details={"expected_path": str(adc_path)},
            ))

    def _check_gcs_access(self) -> None:
        """Check GCS bucket access."""
        bucket_name = os.environ.get("GCS_BUCKET_NAME", "")

        if not bucket_name:
            self.report.add(ValidationResult(
                name="GCS Access",
                status=ValidationStatus.SKIP,
                message="GCS_BUCKET_NAME not set",
            ))
            return

        try:
            from google.cloud import storage  # type: ignore[import-untyped]

            client = storage.Client()
            bucket = client.bucket(bucket_name)

            # Check if bucket exists and is accessible
            if bucket.exists():
                self.report.add(ValidationResult(
                    name="GCS Access",
                    status=ValidationStatus.PASS,
                    message=f"Bucket '{bucket_name}' accessible",
                ))
            else:
                self.report.add(ValidationResult(
                    name="GCS Access",
                    status=ValidationStatus.FAIL,
                    message=f"Bucket '{bucket_name}' not found or not accessible",
                ))
        except ImportError:
            self.report.add(ValidationResult(
                name="GCS Access",
                status=ValidationStatus.SKIP,
                message="google-cloud-storage not installed",
            ))
        except Exception as e:
            self.report.add(ValidationResult(
                name="GCS Access",
                status=ValidationStatus.FAIL,
                message=f"GCS access error: {e}",
            ))

    def _check_vertex_ai_access(self) -> None:
        """Check Vertex AI API access."""
        project_id = os.environ.get("GCP_PROJECT_ID", "")
        region = os.environ.get("GCP_REGION", "us-central1")

        if not project_id:
            self.report.add(ValidationResult(
                name="Vertex AI Access",
                status=ValidationStatus.SKIP,
                message="GCP_PROJECT_ID not set",
            ))
            return

        try:
            from google.cloud import aiplatform  # type: ignore[import-untyped]

            aiplatform.init(project=project_id, location=region)

            # Try to list jobs (lightweight operation)
            jobs = aiplatform.CustomJob.list(filter="state=JOB_STATE_RUNNING")
            job_count = sum(1 for _ in jobs)

            self.report.add(ValidationResult(
                name="Vertex AI Access",
                status=ValidationStatus.PASS,
                message=f"API accessible ({job_count} running jobs)",
                details={"project": project_id, "region": region},
            ))
        except ImportError:
            self.report.add(ValidationResult(
                name="Vertex AI Access",
                status=ValidationStatus.SKIP,
                message="google-cloud-aiplatform not installed",
            ))
        except Exception as e:
            self.report.add(ValidationResult(
                name="Vertex AI Access",
                status=ValidationStatus.FAIL,
                message=f"Vertex AI access error: {e}",
            ))

    def _check_cloud_logging_access(self) -> None:
        """Check Cloud Logging API access."""
        project_id = os.environ.get("GCP_PROJECT_ID", "")

        if not project_id:
            self.report.add(ValidationResult(
                name="Cloud Logging Access",
                status=ValidationStatus.SKIP,
                message="GCP_PROJECT_ID not set",
            ))
            return

        try:
            from google.cloud import logging as cloud_logging  # type: ignore[import-untyped]

            client = cloud_logging.Client(project=project_id)

            # Try to list log entries (lightweight)
            list(client.list_entries(max_results=1))

            self.report.add(ValidationResult(
                name="Cloud Logging Access",
                status=ValidationStatus.PASS,
                message="API accessible",
                details={"project": project_id},
            ))
        except ImportError:
            self.report.add(ValidationResult(
                name="Cloud Logging Access",
                status=ValidationStatus.SKIP,
                message="google-cloud-logging not installed",
            ))
        except Exception as e:
            # Permission errors are common - differentiate
            error_str = str(e).lower()
            if "permission" in error_str or "403" in error_str:
                self.report.add(ValidationResult(
                    name="Cloud Logging Access",
                    status=ValidationStatus.WARN,
                    message="Permission possibly restricted (may still work for writes)",
                    details={"error": str(e)[:100]},
                ))
            else:
                self.report.add(ValidationResult(
                    name="Cloud Logging Access",
                    status=ValidationStatus.FAIL,
                    message=f"Cloud Logging access error: {e}",
                ))


def main() -> int:
    """Run GCP credential validation."""
    logger.info("Starting GCP Credential Validation...")

    validator = GCPCredentialValidator()
    report = validator.validate_all()
    report.print_report()

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
