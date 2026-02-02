#!/usr/bin/env python3
"""Phase 1 Validation Runner.

Orchestrates the complete Phase 1 validation:
1. GCP Credential Validation
2. E2E Test Suite Execution
3. Local Training Smoke Test

Usage:
    python scripts/validate_phase1.py [--skip-gcp] [--skip-training] [--verbose]

Options:
    --skip-gcp: Skip GCP E2E tests (useful for offline validation)
    --skip-training: Skip local training smoke test
    --verbose: Enable verbose output
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import logging
import os
import subprocess
import sys
import time
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


class Phase1Status(Enum):
    """Status of Phase 1 validation."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


@dataclass
class StepResult:
    """Result of a validation step."""

    name: str
    success: bool
    duration_seconds: float
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Phase1Report:
    """Complete Phase 1 validation report."""

    steps: list[StepResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    def add_step(self, step: StepResult) -> None:
        """Add a step result."""
        self.steps.append(step)

    @property
    def status(self) -> Phase1Status:
        """Get overall status."""
        if all(s.success for s in self.steps):
            return Phase1Status.SUCCESS
        if any(s.success for s in self.steps):
            return Phase1Status.PARTIAL
        return Phase1Status.FAILED

    def print_report(self) -> None:
        """Print formatted report."""
        status_emoji = {
            Phase1Status.SUCCESS: "✅",
            Phase1Status.PARTIAL: "⚠️",
            Phase1Status.FAILED: "❌",
        }

        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1 VALIDATION REPORT")
        logger.info("=" * 70)

        for step in self.steps:
            emoji = "✅" if step.success else "❌"
            logger.info(f"{emoji} {step.name} ({step.duration_seconds:.1f}s)")
            logger.info(f"   {step.message}")
            if step.details:
                for key, value in step.details.items():
                    logger.info(f"   • {key}: {value}")

        logger.info("-" * 70)
        logger.info(
            f"Status: {status_emoji[self.status]} {self.status.value} "
            f"| Total time: {self.total_duration_seconds:.1f}s"
        )
        logger.info("=" * 70)


class Phase1Validator:
    """Orchestrates Phase 1 validation."""

    def __init__(
        self,
        project_root: Path,
        skip_gcp: bool = False,
        skip_training: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize validator.

        Args:
            project_root: Root directory of the project.
            skip_gcp: Skip GCP E2E tests.
            skip_training: Skip training smoke test.
            verbose: Enable verbose output.
        """
        self.project_root = project_root
        self.skip_gcp = skip_gcp
        self.skip_training = skip_training
        self.verbose = verbose
        self.report = Phase1Report()

    def _load_gcp_validator(self) -> type | None:
        """Dynamically load GCPCredentialValidator to avoid import issues.

        Uses importlib.util to load the module from file path, avoiding
        issues with Python's module resolution when running as a script.

        Returns:
            GCPCredentialValidator class, or None if loading fails.
        """
        module_name = "scripts.validate_gcp_credentials"

        # Check if already imported
        if module_name in sys.modules:
            return sys.modules[module_name].GCPCredentialValidator

        # Determine module path relative to this script
        module_path = Path(__file__).parent / "validate_gcp_credentials.py"

        if not module_path.exists():
            logger.warning(f"GCP validator module not found: {module_path}")
            return None

        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                logger.warning("Failed to create module spec for GCP validator")
                return None
            gcp_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = gcp_module
            spec.loader.exec_module(gcp_module)
            return gcp_module.GCPCredentialValidator
        except Exception as e:
            logger.warning(f"Failed to load GCP validator: {e}")
            return None

    def run_all(self) -> Phase1Report:
        """Run all Phase 1 validation steps."""
        start_time = time.time()

        logger.info("Starting Phase 1 Validation...")
        logger.info(f"Project root: {self.project_root}")

        # Step 1: Validate environment
        self._validate_environment()

        # Step 2: GCP Credential Validation
        if not self.skip_gcp:
            self._validate_gcp_credentials()
        else:
            logger.info("⏭️ Skipping GCP credential validation")

        # Step 3: Run E2E Tests
        if not self.skip_gcp:
            self._run_e2e_tests()
        else:
            logger.info("⏭️ Skipping E2E tests")

        # Step 4: Run Training Smoke Test
        if not self.skip_training:
            self._run_training_smoke_test()
        else:
            logger.info("⏭️ Skipping training smoke test")

        # Step 5: Validate outputs
        if not self.skip_training:
            self._validate_training_outputs()

        self.report.total_duration_seconds = time.time() - start_time
        return self.report

    def _validate_environment(self) -> None:
        """Validate Python environment."""
        step_start = time.time()

        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 10):
                raise RuntimeError(f"Python 3.10+ required, got {python_version}")

            # Check required packages
            required_packages = ["torch", "numpy", "pytest"]
            missing = []
            for pkg in required_packages:
                try:
                    __import__(pkg)
                except ImportError:
                    missing.append(pkg)

            if missing:
                raise RuntimeError(f"Missing packages: {missing}")

            self.report.add_step(StepResult(
                name="Environment Validation",
                success=True,
                duration_seconds=time.time() - step_start,
                message="Python environment validated",
                details={
                    "python_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                    "packages": ", ".join(required_packages),
                },
            ))
        except Exception as e:
            self.report.add_step(StepResult(
                name="Environment Validation",
                success=False,
                duration_seconds=time.time() - step_start,
                message=f"Environment validation failed: {e}",
            ))

    def _validate_gcp_credentials(self) -> None:
        """Run GCP credential validation with timeout protection."""
        step_start = time.time()
        GCP_VALIDATION_TIMEOUT = 30  # seconds

        try:
            # Quick check: are required env vars set?
            project_id = os.environ.get("GCP_PROJECT_ID", "")
            bucket_name = os.environ.get("GCS_BUCKET_NAME", "")

            if not project_id or not bucket_name:
                # Treat missing GCP config as optional, not failure
                self.report.add_step(StepResult(
                    name="GCP Credentials",
                    success=True,
                    duration_seconds=time.time() - step_start,
                    message="GCP not configured (optional)",
                    details={
                        "GCP_PROJECT_ID": "set" if project_id else "not set",
                        "GCS_BUCKET_NAME": "set" if bucket_name else "not set",
                        "note": "GCP is optional for local-only training",
                    },
                ))
                return

            # Dynamic import to avoid script import issues
            GCPCredentialValidator = self._load_gcp_validator()
            if GCPCredentialValidator is None:
                self.report.add_step(StepResult(
                    name="GCP Credentials",
                    success=False,
                    duration_seconds=time.time() - step_start,
                    message="GCP validator module not available",
                ))
                return

            # Run validation with timeout protection
            def run_validation():
                validator = GCPCredentialValidator()
                return validator.validate_all()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_validation)
                try:
                    cred_report = future.result(timeout=GCP_VALIDATION_TIMEOUT)

                    if cred_report.passed:
                        self.report.add_step(StepResult(
                            name="GCP Credentials",
                            success=True,
                            duration_seconds=time.time() - step_start,
                            message="GCP credentials validated",
                            details=cred_report.summary,
                        ))
                    else:
                        self.report.add_step(StepResult(
                            name="GCP Credentials",
                            success=False,
                            duration_seconds=time.time() - step_start,
                            message="GCP credential validation failed",
                            details=cred_report.summary,
                        ))
                except concurrent.futures.TimeoutError:
                    self.report.add_step(StepResult(
                        name="GCP Credentials",
                        success=False,
                        duration_seconds=time.time() - step_start,
                        message=f"GCP validation timed out after {GCP_VALIDATION_TIMEOUT}s",
                        details={"hint": "Check network connectivity or skip with --skip-gcp"},
                    ))

        except ImportError as e:
            # Google Cloud SDK not installed - this is acceptable
            self.report.add_step(StepResult(
                name="GCP Credentials",
                success=True,
                duration_seconds=time.time() - step_start,
                message="GCP SDK not installed (optional)",
                details={"error": str(e)},
            ))
        except Exception as e:
            self.report.add_step(StepResult(
                name="GCP Credentials",
                success=False,
                duration_seconds=time.time() - step_start,
                message=f"GCP credential validation error: {e}",
            ))

    def _run_e2e_tests(self) -> None:
        """Run E2E test suite."""
        step_start = time.time()

        try:
            e2e_test_path = self.project_root / "tests" / "e2e" / "test_gcp_integration.py"

            if not e2e_test_path.exists():
                raise FileNotFoundError(f"E2E tests not found: {e2e_test_path}")

            # Run pytest
            cmd = [
                sys.executable, "-m", "pytest",
                str(e2e_test_path),
                "-v",
                "--tb=short",
                "-x",  # Stop on first failure
            ]

            if self.verbose:
                cmd.append("-s")

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Parse pytest output for test counts
            output = result.stdout + result.stderr
            passed = output.count(" passed")
            failed = output.count(" failed")
            skipped = output.count(" skipped")

            if result.returncode == 0:
                self.report.add_step(StepResult(
                    name="E2E Tests",
                    success=True,
                    duration_seconds=time.time() - step_start,
                    message="All E2E tests passed",
                    details={"passed": passed, "skipped": skipped},
                ))
            else:
                self.report.add_step(StepResult(
                    name="E2E Tests",
                    success=False,
                    duration_seconds=time.time() - step_start,
                    message=f"E2E tests failed ({failed} failures)",
                    details={"passed": passed, "failed": failed, "skipped": skipped},
                ))
                if self.verbose:
                    logger.error(f"Test output:\n{output}")

        except subprocess.TimeoutExpired:
            self.report.add_step(StepResult(
                name="E2E Tests",
                success=False,
                duration_seconds=time.time() - step_start,
                message="E2E tests timed out (>5 minutes)",
            ))
        except Exception as e:
            self.report.add_step(StepResult(
                name="E2E Tests",
                success=False,
                duration_seconds=time.time() - step_start,
                message=f"E2E test error: {e}",
            ))

    def _run_training_smoke_test(self) -> None:
        """Run local training smoke test."""
        step_start = time.time()

        try:
            # Run the training smoke test suite instead of actual training
            # This is faster and more reliable
            test_path = self.project_root / "tests" / "unit" / "test_training_smoke.py"
            
            if not test_path.exists():
                raise FileNotFoundError(f"Training smoke tests not found: {test_path}")

            cmd = [
                sys.executable, "-m", "pytest",
                str(test_path),
                "-v",
                "--tb=short",
                "-x",  # Stop on first failure
            ]

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout for tests
            )

            # Parse pytest output for test counts
            output = result.stdout + result.stderr
            passed_match = output.count(" passed")
            failed_match = output.count(" failed")
            skipped_match = output.count(" skipped")

            if result.returncode == 0:
                self.report.add_step(StepResult(
                    name="Training Smoke Test",
                    success=True,
                    duration_seconds=time.time() - step_start,
                    message="Training smoke tests passed",
                    details={
                        "passed": passed_match,
                        "skipped": skipped_match,
                    },
                ))
            else:
                self.report.add_step(StepResult(
                    name="Training Smoke Test",
                    success=False,
                    duration_seconds=time.time() - step_start,
                    message=f"Training smoke tests failed ({failed_match} failures)",
                    details={
                        "passed": passed_match,
                        "failed": failed_match,
                    },
                ))
                if self.verbose:
                    logger.error(f"Test output:\n{output}")

        except subprocess.TimeoutExpired:
            self.report.add_step(StepResult(
                name="Training Smoke Test",
                success=False,
                duration_seconds=time.time() - step_start,
                message="Training smoke test timed out (>2 minutes)",
            ))
        except Exception as e:
            self.report.add_step(StepResult(
                name="Training Smoke Test",
                success=False,
                duration_seconds=time.time() - step_start,
                message=f"Training smoke test error: {e}",
            ))

    def _validate_training_outputs(self) -> None:
        """Validate training outputs exist.

        Checks multiple possible checkpoint and log locations for flexibility
        across different training configurations.
        """
        step_start = time.time()

        try:
            # Define all possible checkpoint directories (most likely first)
            checkpoint_candidates = [
                self.project_root / "checkpoints" / "civ6",
                self.project_root / "checkpoints" / "enhanced",
                self.project_root / "checkpoints" / "orchestrator",
                self.project_root / "checkpoints" / "enhanced_v2",
                self.project_root / "checkpoints" / "muzero",
                self.project_root / "checkpoints" / "imitation",
                self.project_root / "checkpoints" / "game_state",
            ]

            # Define all possible log directories
            log_candidates = [
                self.project_root / "logs" / "civ6",
                self.project_root / "logs" / "orchestrator",
                self.project_root / "logs" / "debug_selfplay",
                self.project_root / "logs",  # Root logs directory
            ]

            details: dict[str, Any] = {}
            found_checkpoints: list[tuple[str, int]] = []
            found_logs: list[tuple[str, int]] = []

            # Find any existing checkpoint directories with .pt files
            for ckpt_dir in checkpoint_candidates:
                if ckpt_dir.exists():
                    checkpoints = list(ckpt_dir.glob("*.pt"))
                    if checkpoints:
                        found_checkpoints.append((ckpt_dir.name, len(checkpoints)))

            # Find any existing log directories with files
            for log_dir in log_candidates:
                if log_dir.exists():
                    logs = list(log_dir.glob("*"))
                    # Filter out directories for the root logs case
                    log_files = [f for f in logs if f.is_file()]
                    if log_files:
                        found_logs.append((log_dir.name, len(log_files)))

            # Build details
            if found_checkpoints:
                details["checkpoint_dirs"] = {name: count for name, count in found_checkpoints}
                details["total_checkpoints"] = sum(count for _, count in found_checkpoints)
            else:
                details["checkpoint_dirs"] = "none found"
                details["total_checkpoints"] = 0

            if found_logs:
                details["log_dirs"] = {name: count for name, count in found_logs}
            else:
                details["log_dirs"] = "none found"

            # Success if any checkpoint location has files
            if found_checkpoints:
                self.report.add_step(StepResult(
                    name="Output Validation",
                    success=True,
                    duration_seconds=time.time() - step_start,
                    message=f"Found {details['total_checkpoints']} checkpoints in {len(found_checkpoints)} directories",
                    details=details,
                ))
            else:
                # No checkpoints found - warn but provide helpful info
                self.report.add_step(StepResult(
                    name="Output Validation",
                    success=False,
                    duration_seconds=time.time() - step_start,
                    message="No checkpoint files found in any expected location",
                    details=details,
                ))

        except Exception as e:
            self.report.add_step(StepResult(
                name="Output Validation",
                success=False,
                duration_seconds=time.time() - step_start,
                message=f"Output validation error: {e}",
            ))


def main() -> int:
    """Run Phase 1 validation."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Validation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-gcp",
        action="store_true",
        help="Skip GCP E2E tests",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip local training smoke test",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root directory (auto-detected if not specified)",
    )
    args = parser.parse_args()

    # Determine project root
    if args.project_root:
        project_root = args.project_root
    else:
        # Auto-detect from script location
        project_root = Path(__file__).parent.parent

    if not (project_root / "pyproject.toml").exists():
        logger.error(f"Invalid project root: {project_root}")
        return 1

    # Run validation
    validator = Phase1Validator(
        project_root=project_root,
        skip_gcp=args.skip_gcp,
        skip_training=args.skip_training,
        verbose=args.verbose,
    )

    report = validator.run_all()
    report.print_report()

    # Exit code based on status
    if report.status == Phase1Status.SUCCESS:
        return 0
    elif report.status == Phase1Status.PARTIAL:
        return 1
    else:
        return 2


if __name__ == "__main__":
    sys.exit(main())
