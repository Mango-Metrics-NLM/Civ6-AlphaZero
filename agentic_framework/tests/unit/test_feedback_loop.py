"""
Tests for FeedbackLoop class.
"""
from unittest.mock import MagicMock, patch

import pytest

from agentic_framework.config import AgenticConfig
from agentic_framework.core.feedback_loop import (
    ErrorAction,
    ErrorPattern,
    FeedbackLoop,
    VerificationCommand,
    VerificationPhase,
)
from agentic_framework.utils.command_runner import CommandRunner


class TestFeedbackLoop:
    """Tests for FeedbackLoop."""

    @pytest.fixture
    def mock_command_runner(self):
        """Create a mock command runner."""
        return MagicMock(spec=CommandRunner)

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = MagicMock(spec=AgenticConfig)
        # Create nested mocks for attributes
        config.feedback = MagicMock()
        config.feedback.verification_commands = []
        config.feedback.success_verification = []
        config.feedback.error_handling = MagicMock()
        config.feedback.error_handling.auto_fix_patterns = []
        return config

    def test_initialization(self, mock_config, mock_command_runner):
        """Test initialization with dependency injection."""
        loop = FeedbackLoop(
            config=mock_config,
            command_runner=mock_command_runner,
        )
        assert loop.command_runner == mock_command_runner
        assert loop.config == mock_config

    def test_run_verification_success(self, mock_config, mock_command_runner):
        """Test successful verification run."""
        loop = FeedbackLoop(config=mock_config, command_runner=mock_command_runner)

        command = VerificationCommand(name="test", command="echo test")

        # Mock successful execution
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.stdout = "output"
        mock_result.stderr = ""
        mock_command_runner.run.return_value = mock_result

        result = loop.run_verification(command)

        assert result.success
        assert result.phase == VerificationPhase.CUSTOM
        assert len(result.errors) == 0

    def test_run_verification_failure_with_errors(self, mock_config, mock_command_runner):
        """Test verification failure with error detection."""
        loop = FeedbackLoop(config=mock_config, command_runner=mock_command_runner)

        # Add a custom error pattern
        pattern = ErrorPattern(
            error_type="custom_error",
            action=ErrorAction.REPORT,
            pattern=None  # We'll mock detect_errors anyway, or could use regex
        )
        loop.error_handler.register_pattern(pattern)

        command = VerificationCommand(name="lint", command="lint cmd")

        # Mock failed execution
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.stdout = "Error: something wrong"
        mock_result.stderr = ""
        mock_command_runner.run.return_value = mock_result

        # Mock error detection
        with patch.object(loop.error_handler, 'detect_errors') as mock_detect:
            mock_detect.return_value = [{
                "type": "custom_error",
                "message": "something wrong",
                "action": "report",
                "max_retries": 1
            }]

            result = loop.run_verification(command)

        assert not result.success
        assert len(result.errors) == 1
        assert result.errors[0]["type"] == "custom_error"

    def test_determine_phase(self, mock_config):
        """Test phase determination logic."""
        loop = FeedbackLoop(config=mock_config)

        assert loop._determine_phase("pytest") == VerificationPhase.UNIT_TEST
        assert loop._determine_phase("ruff check") == VerificationPhase.LINT
        assert loop._determine_phase("mypy") == VerificationPhase.TYPE_CHECK
        assert loop._determine_phase("unknown") == VerificationPhase.CUSTOM
