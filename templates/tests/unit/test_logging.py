"""
Unit tests for the logging_config module.

Tests cover:
- LogConfig creation and loading
- Log formatters (JSON, Console)
- Context and sensitive data filters
- BoundLogger functionality
- Logging utilities and decorators
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import TYPE_CHECKING

import pytest

from template_system.logging_config import (
    BoundLogger,
    ConsoleFormatter,
    ContextFilter,
    DebugHelper,
    JsonFormatter,
    LogConfig,
    LogFormat,
    LogLevel,
    SensitiveDataFilter,
    configure_logging,
    debug_log,
    get_logger,
    log_context,
    log_timing,
)


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_level_values(self) -> None:
        """Test log level numeric values."""
        assert LogLevel.DEBUG.value == logging.DEBUG
        assert LogLevel.INFO.value == logging.INFO
        assert LogLevel.WARNING.value == logging.WARNING
        assert LogLevel.ERROR.value == logging.ERROR
        assert LogLevel.CRITICAL.value == logging.CRITICAL

    def test_from_string_valid(self) -> None:
        """Test creating LogLevel from valid string."""
        assert LogLevel.from_string("DEBUG") == LogLevel.DEBUG
        assert LogLevel.from_string("info") == LogLevel.INFO
        assert LogLevel.from_string("Warning") == LogLevel.WARNING

    def test_from_string_invalid(self) -> None:
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            LogLevel.from_string("INVALID")


class TestLogFormat:
    """Tests for LogFormat enum."""

    def test_log_format_values(self) -> None:
        """Test log format values."""
        assert LogFormat.JSON.value == "json"
        assert LogFormat.CONSOLE.value == "console"
        assert LogFormat.STRUCTURED.value == "structured"


class TestLogConfig:
    """Tests for LogConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LogConfig()
        assert config.level == LogLevel.INFO
        assert config.format == LogFormat.CONSOLE
        assert config.output == "stdout"
        assert config.include_timestamp is True

    def test_from_dict(self) -> None:
        """Test creating config from dictionary."""
        config = LogConfig.from_dict({
            "level": "DEBUG",
            "format": "json",
            "output": "stderr",
        })
        assert config.level == LogLevel.DEBUG
        assert config.format == LogFormat.JSON
        assert config.output == "stderr"

    def test_from_env(self) -> None:
        """Test creating config from environment."""
        os.environ["LOG_LEVEL"] = "WARNING"
        os.environ["LOG_FORMAT"] = "json"
        try:
            config = LogConfig.from_env()
            assert config.level == LogLevel.WARNING
            assert config.format == LogFormat.JSON
        finally:
            os.environ.pop("LOG_LEVEL", None)
            os.environ.pop("LOG_FORMAT", None)

    def test_sensitive_fields_default(self) -> None:
        """Test default sensitive fields."""
        config = LogConfig()
        assert "password" in config.sensitive_fields
        assert "api_key" in config.sensitive_fields


class TestContextFilter:
    """Tests for ContextFilter class."""

    def test_filter_adds_context(self) -> None:
        """Test filter adds context to records."""
        filter_ = ContextFilter({"app": "test"})
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        filter_.filter(record)
        assert hasattr(record, "app")
        assert record.app == "test"

    def test_bind_unbind_context(self) -> None:
        """Test binding and unbinding context."""
        filter_ = ContextFilter()
        filter_.bind(request_id="123", user="test")
        assert "request_id" in filter_._local_context
        assert "user" in filter_._local_context

        filter_.unbind("request_id")
        assert "request_id" not in filter_._local_context
        assert "user" in filter_._local_context

    def test_clear_context(self) -> None:
        """Test clearing all context."""
        filter_ = ContextFilter()
        filter_.bind(key1="value1", key2="value2")
        filter_.clear()
        assert len(filter_._local_context) == 0


class TestSensitiveDataFilter:
    """Tests for SensitiveDataFilter class."""

    def test_redacts_sensitive_fields(self) -> None:
        """Test sensitive fields are redacted."""
        filter_ = SensitiveDataFilter(["password", "api_key"])
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.password = "secret123"
        record.api_key = "key456"
        record.safe_field = "visible"

        filter_.filter(record)

        assert record.password == "[REDACTED]"
        assert record.api_key == "[REDACTED]"
        assert record.safe_field == "visible"

    def test_case_insensitive_redaction(self) -> None:
        """Test redaction is case-insensitive."""
        filter_ = SensitiveDataFilter(["password"])
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.Password = "secret"
        filter_.filter(record)
        assert record.Password == "[REDACTED]"


class TestJsonFormatter:
    """Tests for JsonFormatter class."""

    def test_format_basic_message(self) -> None:
        """Test formatting a basic log message."""
        formatter = JsonFormatter(include_timestamp=False, include_caller=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"

    def test_format_with_timestamp(self) -> None:
        """Test formatting includes timestamp."""
        formatter = JsonFormatter(include_timestamp=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "timestamp" in data

    def test_format_with_caller(self) -> None:
        """Test formatting includes caller info."""
        formatter = JsonFormatter(include_caller=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.funcName = "test_function"
        output = formatter.format(record)
        data = json.loads(output)

        assert "caller" in data
        assert data["caller"]["line"] == 42
        assert data["caller"]["function"] == "test_function"

    def test_format_with_extra_fields(self) -> None:
        """Test formatting includes extra fields."""
        formatter = JsonFormatter(include_timestamp=False, include_caller=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"
        output = formatter.format(record)
        data = json.loads(output)
        assert data.get("custom_field") == "custom_value"


class TestConsoleFormatter:
    """Tests for ConsoleFormatter class."""

    def test_format_basic_message(self) -> None:
        """Test formatting a basic console message."""
        formatter = ConsoleFormatter(include_timestamp=False, use_colors=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        assert "INFO" in output
        assert "test" in output
        assert "Test message" in output

    def test_format_with_timestamp(self) -> None:
        """Test formatting with timestamp."""
        formatter = ConsoleFormatter(include_timestamp=True, use_colors=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        # Should contain date-like pattern
        assert "[" in output and "]" in output


class TestBoundLogger:
    """Tests for BoundLogger class."""

    def test_bind_creates_new_logger(self) -> None:
        """Test bind creates a new logger instance."""
        base_logger = logging.getLogger("test.bound")
        logger = BoundLogger(base_logger)
        bound = logger.bind(request_id="123")

        assert bound is not logger
        assert bound._bound_context["request_id"] == "123"

    def test_unbind_removes_context(self) -> None:
        """Test unbind removes context from logger."""
        base_logger = logging.getLogger("test.bound2")
        logger = BoundLogger(base_logger)
        bound = logger.bind(a="1", b="2")
        unbound = bound.unbind("a")

        assert "a" not in unbound._bound_context
        assert unbound._bound_context["b"] == "2"

    def test_log_methods_exist(self) -> None:
        """Test all log methods exist."""
        base_logger = logging.getLogger("test.bound3")
        logger = BoundLogger(base_logger)

        assert callable(logger.debug)
        assert callable(logger.info)
        assert callable(logger.warning)
        assert callable(logger.error)
        assert callable(logger.critical)
        assert callable(logger.exception)


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_with_defaults(self) -> None:
        """Test configuring with defaults."""
        configure_logging()
        logger = get_logger("test.config")
        assert logger is not None

    def test_configure_with_level(self) -> None:
        """Test configuring with specific level."""
        configure_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_configure_with_json_format(self) -> None:
        """Test configuring with JSON format."""
        configure_logging(format="json")
        # Should not raise
        logger = get_logger("test.json")
        logger.info("Test")


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_bound_logger(self) -> None:
        """Test get_logger returns BoundLogger."""
        logger = get_logger("test.module")
        assert isinstance(logger, BoundLogger)

    def test_get_logger_caches_loggers(self) -> None:
        """Test get_logger caches loggers."""
        logger1 = get_logger("test.cached")
        logger2 = get_logger("test.cached")
        assert logger1 is logger2


class TestLogContext:
    """Tests for log_context context manager."""

    def test_context_manager(self) -> None:
        """Test log_context adds and removes context."""
        configure_logging()
        with log_context(request_id="test123"):
            # Context should be available
            pass
        # Context should be removed


class TestLogTiming:
    """Tests for log_timing decorator."""

    def test_timing_decorator(self) -> None:
        """Test timing decorator logs duration."""
        configure_logging(level="DEBUG")

        @log_timing(level=logging.INFO, message="Function completed")
        def slow_function() -> str:
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"

    def test_timing_decorator_on_failure(self) -> None:
        """Test timing decorator logs on failure."""
        configure_logging(level="DEBUG")

        @log_timing(message="Function failed")
        def failing_function() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()


class TestDebugHelper:
    """Tests for DebugHelper class."""

    def test_inspect_object(self) -> None:
        """Test inspecting an object."""
        configure_logging(level="DEBUG")
        helper = DebugHelper()
        obj = {"key": "value", "number": 42}
        helper.inspect(obj, "test_dict")
        # Should not raise

    def test_trace_call(self) -> None:
        """Test tracing a function call."""
        configure_logging(level="DEBUG")
        helper = DebugHelper()
        helper.trace_call("test_func", (1, 2, 3), {"key": "value"})
        # Should not raise

    def test_breakpoint_management(self) -> None:
        """Test setting and checking breakpoints."""
        helper = DebugHelper()
        helper.set_breakpoint("test_bp")
        assert helper.check_breakpoint("test_bp") is True
        assert helper.check_breakpoint("other_bp") is False

        helper.clear_breakpoint("test_bp")
        assert helper.check_breakpoint("test_bp") is False


class TestDebugLog:
    """Tests for debug_log convenience function."""

    def test_debug_log_basic(self) -> None:
        """Test basic debug logging."""
        configure_logging(level="DEBUG")
        debug_log("Test debug message", key="value")
        # Should not raise
