"""
Alert Management System for Civ6 AlphaZero Cloud Training.

Provides unified alerting across multiple channels:
- Console logging
- Slack webhooks
- Email (SMTP)
- W&B alerts

Features:
- Rate limiting to prevent spam
- Alert history tracking
- Configurable severity thresholds per channel
- Callback support for custom handlers

All values configurable via environment variables with CIV6AZ_ALERT_ prefix.
"""

from __future__ import annotations

import logging
import os
import smtplib
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from alpha_zero.cloud.gcp.exceptions import ConfigurationError
from alpha_zero.utils.compat import UTC

if TYPE_CHECKING:
    from alpha_zero.monitoring.mlflow_logger import WandbCompatAdapter as WandbLogger

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(str, Enum):
    """Available alert channels."""

    CONSOLE = "console"
    SLACK = "slack"
    EMAIL = "email"
    WANDB = "wandb"


@dataclass
class AlertEvent:
    """Represents an alert event.

    Attributes:
        alert_id: Unique identifier for this alert.
        severity: Alert severity level.
        title: Short alert title.
        message: Detailed message.
        source: Source identifier (e.g., "budget", "pipeline", "elo").
        timestamp: When alert was created (ISO format).
        details: Additional context details.
        channels_notified: Channels that received this alert.
    """

    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    details: dict[str, Any] = field(default_factory=dict)
    channels_notified: list[AlertChannel] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp,
            "details": self.details,
            "channels_notified": [c.value for c in self.channels_notified],
        }


@dataclass
class SlackConfig:
    """Slack alert configuration."""

    enabled: bool = field(
        default_factory=lambda: (
            os.environ.get("CIV6AZ_ALERT_SLACK_ENABLED", "false").lower() == "true"
        )
    )
    webhook_url: str = field(
        default_factory=lambda: os.environ.get("CIV6AZ_ALERT_SLACK_WEBHOOK", "")
    )
    channel: str = field(
        default_factory=lambda: os.environ.get("CIV6AZ_ALERT_SLACK_CHANNEL", "#civ6-training")
    )
    username: str = "Civ6 AlphaZero Bot"
    icon_emoji: str = ":robot_face:"

    # Severity mapping to Slack colors
    severity_colors: dict[str, str] = field(
        default_factory=lambda: {
            "info": "#36a64f",  # Green
            "warning": "#ffc107",  # Yellow
            "critical": "#dc3545",  # Red
            "emergency": "#7b1fa2",  # Purple
        }
    )


@dataclass
class EmailConfig:
    """Email alert configuration."""

    enabled: bool = field(
        default_factory=lambda: (
            os.environ.get("CIV6AZ_ALERT_EMAIL_ENABLED", "false").lower() == "true"
        )
    )
    smtp_host: str = field(
        default_factory=lambda: os.environ.get("CIV6AZ_ALERT_SMTP_HOST", "smtp.gmail.com")
    )
    smtp_port: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_ALERT_SMTP_PORT", "587"))
    )
    smtp_user: str = field(default_factory=lambda: os.environ.get("CIV6AZ_ALERT_SMTP_USER", ""))
    smtp_password: str = field(
        default_factory=lambda: os.environ.get("CIV6AZ_ALERT_SMTP_PASSWORD", "")
    )
    from_address: str = field(default_factory=lambda: os.environ.get("CIV6AZ_ALERT_FROM_EMAIL", ""))
    to_addresses: list[str] = field(
        default_factory=lambda: [
            addr.strip()
            for addr in os.environ.get("CIV6AZ_ALERT_TO_EMAILS", "").split(",")
            if addr.strip()
        ]
    )
    min_severity: AlertSeverity = AlertSeverity.WARNING


@dataclass
class AlertManagerConfig:
    """Configuration for AlertManager."""

    slack: SlackConfig = field(default_factory=SlackConfig)
    email: EmailConfig = field(default_factory=EmailConfig)

    # W&B integration
    wandb_enabled: bool = field(
        default_factory=lambda: (
            os.environ.get("CIV6AZ_ALERT_WANDB_ENABLED", "true").lower() == "true"
        )
    )

    # Console logging
    console_enabled: bool = True

    # Rate limiting to prevent alert spam
    rate_limit_seconds: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_ALERT_RATE_LIMIT", "300"))
    )

    # Alert history retention
    history_max_size: int = 1000

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values."""
        errors: list[str] = []

        if self.rate_limit_seconds < 0:
            errors.append("rate_limit_seconds cannot be negative")

        if self.history_max_size < 1:
            errors.append("history_max_size must be at least 1")

        if errors:
            raise ConfigurationError(
                f"Invalid alert manager configuration: {'; '.join(errors)}",
                details={"errors": errors},
            )

    @classmethod
    def from_env(cls) -> AlertManagerConfig:
        """Create from environment variables."""
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> AlertManagerConfig:
        """Load from YAML file."""
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Alert manager config file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {path}: {e}") from e

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlertManagerConfig:
        """Create from dictionary."""
        slack_data = data.get("slack", {})
        email_data = data.get("email", {})

        return cls(
            slack=SlackConfig(**slack_data) if slack_data else SlackConfig(),
            email=EmailConfig(**email_data) if email_data else EmailConfig(),
            wandb_enabled=data.get("wandb_enabled", True),
            console_enabled=data.get("console_enabled", True),
            rate_limit_seconds=data.get("rate_limit_seconds", 300),
            history_max_size=data.get("history_max_size", 1000),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "wandb_enabled": self.wandb_enabled,
            "console_enabled": self.console_enabled,
            "rate_limit_seconds": self.rate_limit_seconds,
            "history_max_size": self.history_max_size,
            "slack": {
                "enabled": self.slack.enabled,
                "channel": self.slack.channel,
            },
            "email": {
                "enabled": self.email.enabled,
                "smtp_host": self.email.smtp_host,
                "min_severity": self.email.min_severity.value,
            },
        }


class AlertManager:
    """Unified alert management system.

    Handles sending alerts across multiple channels with:
    - Rate limiting to prevent spam
    - Alert history tracking
    - Configurable severity thresholds per channel
    - Callback support for custom handlers

    Usage:
        config = AlertManagerConfig()
        manager = AlertManager(config)

        # Send an alert
        manager.send_alert(
            severity=AlertSeverity.WARNING,
            title="Budget Warning",
            message="80% of monthly budget consumed",
            source="budget",
            details={"current_spend": 520, "budget": 650},
        )

        # Register custom callback
        manager.register_callback(my_custom_handler)
    """

    def __init__(
        self,
        config: AlertManagerConfig | None = None,
        wandb_logger: WandbLogger | None = None,
    ) -> None:
        """Initialize AlertManager.

        Args:
            config: Configuration (defaults to from env).
            wandb_logger: Optional W&B logger for alerts.
        """
        self.config = config or AlertManagerConfig()
        self._wandb_logger = wandb_logger

        # Alert history
        self._history: deque[AlertEvent] = deque(maxlen=self.config.history_max_size)
        self._last_alert_times: dict[str, float] = {}  # source -> last alert time

        # Custom callbacks
        self._callbacks: list[Callable[[AlertEvent], None]] = []

        # Thread safety
        self._lock = threading.Lock()

        # Alert counter for ID generation
        self._alert_counter = 0

        logger.info(
            f"AlertManager initialized: "
            f"slack={self.config.slack.enabled}, "
            f"email={self.config.email.enabled}, "
            f"wandb={self.config.wandb_enabled}"
        )

    def send_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        details: dict[str, Any] | None = None,
        force: bool = False,
    ) -> AlertEvent | None:
        """Send an alert across configured channels.

        Args:
            severity: Alert severity level.
            title: Short alert title.
            message: Detailed message.
            source: Source identifier (budget, pipeline, elo).
            details: Additional context details.
            force: Bypass rate limiting.

        Returns:
            AlertEvent if sent, None if rate-limited.
        """
        # Rate limiting check
        if not force and not self._should_send(source):
            logger.debug(f"Alert rate-limited: {source}")
            return None

        # Create alert event
        with self._lock:
            self._alert_counter += 1
            alert_id = (
                f"{source}-{self._alert_counter}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
            )

        event = AlertEvent(
            alert_id=alert_id,
            severity=severity,
            title=title,
            message=message,
            source=source,
            details=details or {},
        )

        # Send to channels
        if self.config.console_enabled:
            self._send_console(event)
            event.channels_notified.append(AlertChannel.CONSOLE)

        if self.config.slack.enabled:
            if self._send_slack(event):
                event.channels_notified.append(AlertChannel.SLACK)

        if self.config.email.enabled:
            severity_order = [
                AlertSeverity.INFO,
                AlertSeverity.WARNING,
                AlertSeverity.CRITICAL,
                AlertSeverity.EMERGENCY,
            ]
            if severity_order.index(severity) >= severity_order.index(
                self.config.email.min_severity
            ):
                if self._send_email(event):
                    event.channels_notified.append(AlertChannel.EMAIL)

        if self.config.wandb_enabled and self._wandb_logger:
            if self._send_wandb(event):
                event.channels_notified.append(AlertChannel.WANDB)

        # Invoke callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        # Record in history
        with self._lock:
            self._history.append(event)
            self._last_alert_times[source] = time.time()

        return event

    def _should_send(self, source: str) -> bool:
        """Check if we should send based on rate limiting."""
        with self._lock:
            last_time = self._last_alert_times.get(source, 0)
            return time.time() - last_time >= self.config.rate_limit_seconds

    def _send_console(self, event: AlertEvent) -> None:
        """Log alert to console."""
        level_map = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR,
            AlertSeverity.EMERGENCY: logging.CRITICAL,
        }
        level = level_map.get(event.severity, logging.INFO)
        logger.log(
            level,
            f"[{event.severity.value.upper()}] {event.title}: {event.message}",
        )

    def _send_slack(self, event: AlertEvent) -> bool:
        """Send alert to Slack."""
        if not self.config.slack.webhook_url:
            return False

        try:
            import requests

            color = self.config.slack.severity_colors.get(event.severity.value, "#000000")

            fields = [
                {"title": k, "value": str(v), "short": True}
                for k, v in list(event.details.items())[:10]  # Limit fields
            ]

            payload = {
                "username": self.config.slack.username,
                "icon_emoji": self.config.slack.icon_emoji,
                "channel": self.config.slack.channel,
                "attachments": [
                    {
                        "color": color,
                        "title": f"[{event.severity.value.upper()}] {event.title}",
                        "text": event.message,
                        "fields": fields,
                        "footer": f"Source: {event.source} | {event.timestamp}",
                    }
                ],
            }

            response = requests.post(
                self.config.slack.webhook_url,
                json=payload,
                timeout=10,
            )
            return response.status_code == 200

        except ImportError:
            logger.warning("requests library not installed, Slack alerts disabled")
            return False
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")
            return False

    def _send_email(self, event: AlertEvent) -> bool:
        """Send alert via email."""
        if not self.config.email.to_addresses:
            return False

        try:
            subject = f"[Civ6 AlphaZero {event.severity.value.upper()}] {event.title}"

            details_str = "\n".join(f"  {k}: {v}" for k, v in event.details.items())

            body = f"""Alert: {event.title}
Severity: {event.severity.value}
Source: {event.source}
Time: {event.timestamp}

Message:
{event.message}

Details:
{details_str if details_str else "  (none)"}
"""

            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = self.config.email.from_address
            msg["To"] = ", ".join(self.config.email.to_addresses)

            with smtplib.SMTP(self.config.email.smtp_host, self.config.email.smtp_port) as server:
                server.starttls()
                server.login(self.config.email.smtp_user, self.config.email.smtp_password)
                server.send_message(msg)

            return True

        except Exception as e:
            logger.error(f"Email alert failed: {e}")
            return False

    def _send_wandb(self, event: AlertEvent) -> bool:
        """Send alert to W&B."""
        if not self._wandb_logger:
            return False

        try:
            # Check if logger is initialized
            if hasattr(self._wandb_logger, "is_initialized"):
                if not self._wandb_logger.is_initialized:
                    return False

            level_map = {
                AlertSeverity.INFO: "INFO",
                AlertSeverity.WARNING: "WARN",
                AlertSeverity.CRITICAL: "ERROR",
                AlertSeverity.EMERGENCY: "ERROR",
            }

            if hasattr(self._wandb_logger, "alert"):
                self._wandb_logger.alert(
                    title=event.title,
                    text=event.message,
                    level=level_map.get(event.severity, "INFO"),
                )
                return True

            return False

        except Exception as e:
            logger.error(f"W&B alert failed: {e}")
            return False

    def register_callback(self, callback: Callable[[AlertEvent], None]) -> None:
        """Register callback for alerts.

        Args:
            callback: Function to call for each alert.
        """
        self._callbacks.append(callback)

    def get_history(
        self,
        source: str | None = None,
        severity: AlertSeverity | None = None,
        limit: int = 100,
    ) -> list[AlertEvent]:
        """Get alert history with optional filtering.

        Args:
            source: Filter by source.
            severity: Filter by severity.
            limit: Maximum number of alerts to return.

        Returns:
            List of AlertEvent objects.
        """
        with self._lock:
            filtered = list(self._history)

        if source:
            filtered = [e for e in filtered if e.source == source]
        if severity:
            filtered = [e for e in filtered if e.severity == severity]

        return filtered[-limit:]

    # =========================================================================
    # Convenience methods for common alert types
    # =========================================================================

    def budget_warning(
        self,
        current_spend: float,
        budget: float,
        percentage: float,
    ) -> AlertEvent | None:
        """Send budget warning alert.

        Args:
            current_spend: Current spend in USD.
            budget: Budget limit in USD.
            percentage: Percentage of budget used.

        Returns:
            AlertEvent if sent.
        """
        return self.send_alert(
            severity=AlertSeverity.WARNING,
            title="Budget Warning",
            message=(
                f"Reached {percentage:.0%} of monthly budget (${current_spend:.2f}/${budget:.2f})"
            ),
            source="budget",
            details={
                "current_spend_usd": round(current_spend, 2),
                "budget_usd": round(budget, 2),
                "percentage": round(percentage, 2),
            },
        )

    def budget_exceeded(
        self,
        current_spend: float,
        budget: float,
    ) -> AlertEvent | None:
        """Send budget exceeded alert.

        Args:
            current_spend: Current spend in USD.
            budget: Budget limit in USD.

        Returns:
            AlertEvent if sent.
        """
        return self.send_alert(
            severity=AlertSeverity.CRITICAL,
            title="Budget Exceeded",
            message=f"Monthly budget exceeded! ${current_spend:.2f}/${budget:.2f}",
            source="budget",
            details={
                "current_spend_usd": round(current_spend, 2),
                "budget_usd": round(budget, 2),
            },
            force=True,  # Always send budget exceeded
        )

    def pipeline_failure(
        self,
        component: str,
        error: str,
        job_id: str | None = None,
    ) -> AlertEvent | None:
        """Send pipeline failure alert.

        Args:
            component: Failed component name.
            error: Error message.
            job_id: Optional job identifier.

        Returns:
            AlertEvent if sent.
        """
        return self.send_alert(
            severity=AlertSeverity.CRITICAL,
            title=f"Pipeline Failure: {component}",
            message=error,
            source="pipeline",
            details={
                "component": component,
                "job_id": job_id,
                "error": error[:500],  # Truncate long errors
            },
            force=True,
        )

    def elo_plateau(
        self,
        current_elo: float,
        iterations_stalled: int,
        threshold: int,
    ) -> AlertEvent | None:
        """Send Elo plateau alert.

        Args:
            current_elo: Current Elo rating.
            iterations_stalled: Number of iterations without improvement.
            threshold: Threshold that triggered alert.

        Returns:
            AlertEvent if sent.
        """
        return self.send_alert(
            severity=AlertSeverity.WARNING,
            title="Elo Rating Plateau Detected",
            message=(
                f"Elo rating has plateaued at {current_elo:.0f} for "
                f"{iterations_stalled} iterations (threshold: {threshold})"
            ),
            source="elo",
            details={
                "current_elo": round(current_elo, 1),
                "iterations_stalled": iterations_stalled,
                "threshold": threshold,
            },
        )

    def training_complete(
        self,
        iterations: int,
        final_elo: float,
        total_games: int,
    ) -> AlertEvent | None:
        """Send training complete alert.

        Args:
            iterations: Total training iterations.
            final_elo: Final Elo rating.
            total_games: Total games played.

        Returns:
            AlertEvent if sent.
        """
        return self.send_alert(
            severity=AlertSeverity.INFO,
            title="Training Complete",
            message=(
                f"Training completed: {iterations} iterations, "
                f"Elo={final_elo:.0f}, {total_games} games"
            ),
            source="training",
            details={
                "iterations": iterations,
                "final_elo": round(final_elo, 1),
                "total_games": total_games,
            },
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get alert manager statistics.

        Returns:
            Dictionary with statistics.
        """
        with self._lock:
            by_source: dict[str, int] = {}
            by_severity: dict[str, int] = {}

            for event in self._history:
                by_source[event.source] = by_source.get(event.source, 0) + 1
                by_severity[event.severity.value] = by_severity.get(event.severity.value, 0) + 1

        return {
            "total_alerts": len(self._history),
            "by_source": by_source,
            "by_severity": by_severity,
            "channels_enabled": {
                "console": self.config.console_enabled,
                "slack": self.config.slack.enabled,
                "email": self.config.email.enabled,
                "wandb": self.config.wandb_enabled,
            },
            "rate_limit_seconds": self.config.rate_limit_seconds,
        }


def get_alert_manager(
    wandb_logger: WandbLogger | None = None,
) -> AlertManager:
    """Get an alert manager instance.

    Convenience function for getting an alert manager.

    Args:
        wandb_logger: Optional W&B logger.

    Returns:
        AlertManager instance.
    """
    return AlertManager(AlertManagerConfig.from_env(), wandb_logger=wandb_logger)
