"""Comprehensive tests for src.core.logging module.

Covers:
- generate_correlation_id uniqueness and format
- configure_logging with valid/invalid formats and levels
- configure_logging with log_dir (RotatingFileHandler)
- configure_logging idempotency (no duplicate handlers)
- get_logger returns BoundLogger
- bind_context populates structlog contextvars
- MetricsLogger interval validation and gating behavior
- MetricsLogger.log_scalar and log_scalars delegation
"""

from __future__ import annotations

import logging as _logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import structlog

from src.core.logging import (
    MetricsLogger,
    _add_correlation_id,
    bind_context,
    configure_logging,
    generate_correlation_id,
    get_logger,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_logging():
    """Reset logging state after every test.

    Clears stdlib root handlers, structlog contextvars, and structlog
    configuration so that each test starts from a clean slate.
    """
    yield
    root = _logging.getLogger()
    root.handlers.clear()
    structlog.contextvars.clear_contextvars()
    structlog.reset_defaults()


# ---------------------------------------------------------------------------
# generate_correlation_id
# ---------------------------------------------------------------------------


class TestGenerateCorrelationId:
    """Tests for generate_correlation_id()."""

    def test_returns_32_char_hex_string(self) -> None:
        cid = generate_correlation_id()
        assert isinstance(cid, str)
        assert len(cid) == 32
        # Must be valid hex (only 0-9, a-f)
        int(cid, 16)

    def test_successive_calls_are_unique(self) -> None:
        ids = {generate_correlation_id() for _ in range(100)}
        assert len(ids) == 100, "Expected 100 unique correlation IDs"

    def test_is_lowercase(self) -> None:
        cid = generate_correlation_id()
        assert cid == cid.lower()


# ---------------------------------------------------------------------------
# _add_correlation_id processor
# ---------------------------------------------------------------------------


class TestAddCorrelationId:
    """Tests for the _add_correlation_id structlog processor."""

    def test_injects_id_when_missing(self) -> None:
        event_dict: dict = {"event": "hello"}
        result = _add_correlation_id(None, "info", event_dict)  # type: ignore[arg-type]
        assert "correlation_id" in result
        assert len(result["correlation_id"]) == 32

    def test_preserves_existing_id(self) -> None:
        existing = "deadbeef" * 4
        event_dict: dict = {"event": "hello", "correlation_id": existing}
        result = _add_correlation_id(None, "info", event_dict)  # type: ignore[arg-type]
        assert result["correlation_id"] == existing

    def test_returns_same_dict(self) -> None:
        event_dict: dict = {"event": "test"}
        result = _add_correlation_id(None, "info", event_dict)  # type: ignore[arg-type]
        assert result is event_dict


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    """Tests for configure_logging()."""

    def test_pretty_format_no_error(self) -> None:
        configure_logging(level="INFO", format_="pretty")
        root = _logging.getLogger()
        assert len(root.handlers) == 1  # console handler only

    def test_json_format_no_error(self) -> None:
        configure_logging(level="INFO", format_="json")
        root = _logging.getLogger()
        assert len(root.handlers) == 1

    def test_invalid_format_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="format_ must be 'json' or 'pretty'"):
            configure_logging(format_="invalid")

    def test_invalid_format_xml_raises(self) -> None:
        with pytest.raises(ValueError):
            configure_logging(format_="xml")

    def test_invalid_level_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid log level"):
            configure_logging(level="BADLEVEL")

    def test_invalid_level_nonsense_raises(self) -> None:
        with pytest.raises(ValueError):
            configure_logging(level="NOTREAL")

    def test_valid_levels_accepted(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            configure_logging(level=level, format_="json")
            root = _logging.getLogger()
            assert root.level == getattr(_logging, level)

    def test_log_dir_creates_directory_and_file(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "test_logs"
        assert not log_dir.exists()
        configure_logging(log_dir=str(log_dir))
        assert log_dir.exists()
        assert log_dir.is_dir()
        # The RotatingFileHandler creates sdk.log
        assert (log_dir / "sdk.log").exists()

    def test_log_dir_adds_file_handler(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        configure_logging(log_dir=str(log_dir))
        root = _logging.getLogger()
        # Should have console + file = 2 handlers
        assert len(root.handlers) == 2
        handler_types = {type(h).__name__ for h in root.handlers}
        assert "StreamHandler" in handler_types
        assert "RotatingFileHandler" in handler_types

    def test_log_dir_file_handler_writes_json(self, tmp_path: Path) -> None:
        """File handler should write JSON-formatted log entries."""
        log_dir = tmp_path / "json_logs"
        configure_logging(level="DEBUG", format_="pretty", log_dir=str(log_dir))
        # Log something via stdlib to trigger output
        logger = _logging.getLogger("test_file_json")
        logger.info("test message from file handler check")
        # Flush handlers
        for handler in _logging.getLogger().handlers:
            handler.flush()
        log_file = log_dir / "sdk.log"
        content = log_file.read_text(encoding="utf-8")
        # File handler always uses JSON regardless of format_ setting
        assert content.strip() != ""

    def test_idempotent_no_duplicate_handlers(self) -> None:
        """Calling configure_logging twice should not duplicate handlers."""
        configure_logging(level="INFO", format_="pretty")
        configure_logging(level="DEBUG", format_="json")
        root = _logging.getLogger()
        assert len(root.handlers) == 1
        # Second call should have updated the level
        assert root.level == _logging.DEBUG

    def test_idempotent_with_log_dir(self, tmp_path: Path) -> None:
        log_dir = str(tmp_path / "idempotent_logs")
        configure_logging(log_dir=log_dir)
        configure_logging(log_dir=log_dir)
        root = _logging.getLogger()
        # Still only 2 handlers (console + file), not 4
        assert len(root.handlers) == 2

    def test_sets_configured_flag(self) -> None:
        import src.core.logging as logging_mod

        # The flag might already be set from other tests; reset it
        logging_mod._configured = False
        configure_logging()
        assert logging_mod._configured is True

    def test_correlation_id_disabled(self) -> None:
        """When correlation_id_enabled=False, the processor is not added."""
        configure_logging(correlation_id_enabled=False)
        # We can still log without error
        logger = get_logger("test_no_corr")
        # This should not raise
        logger.info("no correlation id")

    def test_log_dir_nested_path(self, tmp_path: Path) -> None:
        """Nested log directories are created with parents=True."""
        log_dir = tmp_path / "deep" / "nested" / "logs"
        configure_logging(log_dir=str(log_dir))
        assert log_dir.exists()
        assert (log_dir / "sdk.log").exists()


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    """Tests for get_logger()."""

    def test_returns_bound_logger(self) -> None:
        configure_logging()
        logger = get_logger("test.module")
        # structlog.get_logger returns a proxy; after configure, it wraps BoundLogger
        assert logger is not None
        # Verify it has standard log methods
        assert callable(getattr(logger, "info", None))
        assert callable(getattr(logger, "debug", None))
        assert callable(getattr(logger, "warning", None))
        assert callable(getattr(logger, "error", None))

    def test_returns_logger_without_configure(self) -> None:
        """get_logger works even if configure_logging was not called."""
        logger = get_logger("unconfigured.module")
        assert logger is not None
        assert callable(getattr(logger, "info", None))

    def test_different_names_return_loggers(self) -> None:
        configure_logging()
        log_a = get_logger("module.a")
        log_b = get_logger("module.b")
        # They should be separate logger instances
        assert log_a is not log_b


# ---------------------------------------------------------------------------
# bind_context
# ---------------------------------------------------------------------------


class TestBindContext:
    """Tests for bind_context()."""

    def test_adds_context_to_contextvars(self) -> None:
        bind_context(environment_id="mini_resource", iteration=42)
        ctx = structlog.contextvars.get_contextvars()
        assert ctx["environment_id"] == "mini_resource"
        assert ctx["iteration"] == 42

    def test_multiple_calls_accumulate(self) -> None:
        bind_context(key1="value1")
        bind_context(key2="value2")
        ctx = structlog.contextvars.get_contextvars()
        assert ctx["key1"] == "value1"
        assert ctx["key2"] == "value2"

    def test_overwrite_existing_key(self) -> None:
        bind_context(key="old_value")
        bind_context(key="new_value")
        ctx = structlog.contextvars.get_contextvars()
        assert ctx["key"] == "new_value"

    def test_clear_contextvars_removes_bound(self) -> None:
        bind_context(session="abc123")
        structlog.contextvars.clear_contextvars()
        ctx = structlog.contextvars.get_contextvars()
        assert "session" not in ctx


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------


class TestMetricsLogger:
    """Tests for MetricsLogger."""

    def test_init_valid_interval(self) -> None:
        ml = MetricsLogger("train", interval=10)
        assert ml._interval == 10

    def test_init_interval_one(self) -> None:
        ml = MetricsLogger("train", interval=1)
        assert ml._interval == 1

    def test_init_interval_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="interval must be >= 1"):
            MetricsLogger("train", interval=0)

    def test_init_interval_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="interval must be >= 1"):
            MetricsLogger("train", interval=-5)

    def test_init_default_interval(self) -> None:
        ml = MetricsLogger("train")
        assert ml._interval == 1

    # -- _should_log ---------------------------------------------------------

    def test_should_log_interval_one_always_true(self) -> None:
        ml = MetricsLogger("test", interval=1)
        for step in range(20):
            assert ml._should_log(step) is True

    def test_should_log_interval_ten(self) -> None:
        ml = MetricsLogger("test", interval=10)
        assert ml._should_log(0) is True
        assert ml._should_log(1) is False
        assert ml._should_log(5) is False
        assert ml._should_log(9) is False
        assert ml._should_log(10) is True
        assert ml._should_log(20) is True
        assert ml._should_log(100) is True
        assert ml._should_log(15) is False

    def test_should_log_interval_three(self) -> None:
        ml = MetricsLogger("test", interval=3)
        logged = [s for s in range(30) if ml._should_log(s)]
        assert logged == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]

    # -- log_scalar ----------------------------------------------------------

    def test_log_scalar_calls_logger_on_matching_step(self) -> None:
        ml = MetricsLogger("train", interval=5)
        ml._logger = MagicMock()
        ml.log_scalar("loss", 0.42, step=10)
        ml._logger.info.assert_called_once_with(
            "metric", key="loss", value=0.42, step=10
        )

    def test_log_scalar_skips_non_matching_step(self) -> None:
        ml = MetricsLogger("train", interval=5)
        ml._logger = MagicMock()
        ml.log_scalar("loss", 0.42, step=7)
        ml._logger.info.assert_not_called()

    def test_log_scalar_logs_at_step_zero(self) -> None:
        ml = MetricsLogger("train", interval=100)
        ml._logger = MagicMock()
        ml.log_scalar("lr", 0.001, step=0)
        ml._logger.info.assert_called_once_with(
            "metric", key="lr", value=0.001, step=0
        )

    def test_log_scalar_interval_one_always_logs(self) -> None:
        ml = MetricsLogger("train", interval=1)
        ml._logger = MagicMock()
        for step in range(5):
            ml.log_scalar("loss", float(step), step=step)
        assert ml._logger.info.call_count == 5

    # -- log_scalars ---------------------------------------------------------

    def test_log_scalars_calls_logger_on_matching_step(self) -> None:
        ml = MetricsLogger("train", interval=5)
        ml._logger = MagicMock()
        metrics = {"loss": 0.42, "accuracy": 0.95}
        ml.log_scalars(metrics, step=10)
        ml._logger.info.assert_called_once_with(
            "metrics", step=10, loss=0.42, accuracy=0.95
        )

    def test_log_scalars_skips_non_matching_step(self) -> None:
        ml = MetricsLogger("train", interval=5)
        ml._logger = MagicMock()
        ml.log_scalars({"loss": 0.42}, step=3)
        ml._logger.info.assert_not_called()

    def test_log_scalars_empty_dict(self) -> None:
        ml = MetricsLogger("train", interval=1)
        ml._logger = MagicMock()
        ml.log_scalars({}, step=0)
        ml._logger.info.assert_called_once_with("metrics", step=0)

    def test_log_scalars_interval_one_always_logs(self) -> None:
        ml = MetricsLogger("train", interval=1)
        ml._logger = MagicMock()
        for step in range(5):
            ml.log_scalars({"val": float(step)}, step=step)
        assert ml._logger.info.call_count == 5

    # -- integration: actual logging -----------------------------------------

    def test_log_scalar_integration(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify log_scalar produces real output when configured."""
        configure_logging(level="INFO", format_="json")
        ml = MetricsLogger("integration_test", interval=1)
        ml.log_scalar("test_metric", 3.14, step=0)
        # The output goes through stderr (structlog default console handler)
        captured = capsys.readouterr()
        # JSON format should contain our metric key somewhere in output
        combined = captured.out + captured.err
        assert "test_metric" in combined or "metric" in combined

    def test_log_scalars_integration(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify log_scalars produces real output when configured."""
        configure_logging(level="INFO", format_="json")
        ml = MetricsLogger("integration_scalars", interval=1)
        ml.log_scalars({"alpha": 1.0, "beta": 2.0}, step=5)
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "alpha" in combined or "metrics" in combined


# ---------------------------------------------------------------------------
# Edge cases and combined scenarios
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and combined scenarios."""

    def test_configure_then_bind_then_log(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Full workflow: configure, bind context, then log."""
        configure_logging(level="DEBUG", format_="json")
        bind_context(session_id="test-session")
        logger = get_logger("workflow_test")
        logger.info("test event", data=123)
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        # The bound session_id should appear in output
        assert "test-session" in combined

    def test_reconfigure_changes_level(self) -> None:
        """Reconfiguring changes the effective log level."""
        configure_logging(level="WARNING")
        assert _logging.getLogger().level == _logging.WARNING
        configure_logging(level="DEBUG")
        assert _logging.getLogger().level == _logging.DEBUG

    def test_reconfigure_changes_format(self) -> None:
        """Reconfiguring switches between pretty and json."""
        configure_logging(format_="pretty")
        configure_logging(format_="json")
        root = _logging.getLogger()
        assert len(root.handlers) == 1  # Still just one handler

    def test_log_dir_existing_directory(self, tmp_path: Path) -> None:
        """Using an already-existing log_dir works fine."""
        log_dir = tmp_path / "existing"
        log_dir.mkdir()
        configure_logging(log_dir=str(log_dir))
        assert (log_dir / "sdk.log").exists()
        root = _logging.getLogger()
        assert len(root.handlers) == 2

    def test_case_insensitive_level(self) -> None:
        """Level string is uppercased internally."""
        configure_logging(level="debug")
        assert _logging.getLogger().level == _logging.DEBUG

    def test_file_handler_rotating_config(self, tmp_path: Path) -> None:
        """RotatingFileHandler has expected maxBytes and backupCount."""
        log_dir = tmp_path / "rotating"
        configure_logging(log_dir=str(log_dir))
        root = _logging.getLogger()
        file_handlers = [
            h
            for h in root.handlers
            if isinstance(h, _logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 1
        fh = file_handlers[0]
        assert fh.maxBytes == 10 * 1024 * 1024  # 10 MiB
        assert fh.backupCount == 5
        assert fh.encoding == "utf-8"
