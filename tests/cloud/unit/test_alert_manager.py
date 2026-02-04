"""Unit tests for alert_manager module."""
from __future__ import annotations
import os
import pytest
from alpha_zero.cloud.gcp.alert_manager import (
    AlertChannel, AlertEvent, AlertManager, AlertManagerConfig,
    AlertSeverity, EmailConfig, SlackConfig, get_alert_manager,
)

class TestAlertSeverity:
    def test_severity_values(self) -> None:
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"

class TestSlackConfig:
    def test_default_values(self) -> None:
        config = SlackConfig()
        assert config.enabled is False
        assert config.webhook_url == ""  # Empty string, not None

class TestAlertManagerConfig:
    def test_default_values(self) -> None:
        config = AlertManagerConfig()
        assert isinstance(config.slack, SlackConfig)
        assert config.wandb_enabled is True

class TestAlertManager:
    def test_init(self) -> None:
        config = AlertManagerConfig()
        manager = AlertManager(config)
        assert manager.config == config

    def test_send_alert_console(self) -> None:
        config = AlertManagerConfig(console_enabled=True)
        manager = AlertManager(config)
        event = manager.send_alert(
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test message",
            source="test",
        )
        assert event is not None
        assert event.severity == AlertSeverity.INFO

    def test_convenience_methods(self) -> None:
        config = AlertManagerConfig(console_enabled=True, rate_limit_seconds=0)
        manager = AlertManager(config)
        event = manager.budget_warning(
            current_spend=500.0, budget=650.0, percentage=77.0,
        )
        assert event is not None

    def test_get_statistics(self) -> None:
        config = AlertManagerConfig(console_enabled=True, rate_limit_seconds=0)
        manager = AlertManager(config)
        manager.send_alert(severity=AlertSeverity.INFO, title="Test", message="Test", source="test")
        stats = manager.get_statistics()
        assert "total_alerts" in stats

class TestGetAlertManager:
    def test_returns_manager(self) -> None:
        manager = get_alert_manager()
        assert isinstance(manager, AlertManager)