"""Loopback helpers for the local-first network posture (LF-G14).

The web interface and the inference server bind to loopback by default so a
workstation run is not exposed on the LAN. Binding elsewhere stays possible
(containers legitimately bind all interfaces) but is always logged.
"""

from __future__ import annotations

import ipaddress
import logging
import os

LOOPBACK_HOST = "127.0.0.1"


def is_loopback_host(host: str) -> bool:
    """Return True if ``host`` is ``localhost`` or a loopback IP literal."""
    candidate = host.strip().strip("[]").lower()
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def loopback_origins(port: int) -> list[str]:
    """Return the CORS origins for a loopback-served UI on ``port``."""
    return [f"http://{h}:{port}" for h in (LOOPBACK_HOST, "localhost", "[::1]")]


def warn_if_non_loopback(host: str, logger: logging.Logger, what: str) -> None:
    """Log a WARNING when ``host`` would expose ``what`` beyond this machine."""
    if not is_loopback_host(host):
        logger.warning(
            "%s is bound to non-loopback host %r and is reachable from other machines",
            what,
            host,
        )


def resolve_bind_host(env_var: str, logger: logging.Logger, what: str) -> str:
    """Return the bind address from ``env_var``, defaulting to loopback.

    A non-loopback value is honoured (containers bind all interfaces) but warned.
    """
    host = os.environ.get(env_var, "").strip() or LOOPBACK_HOST
    warn_if_non_loopback(host, logger, what)
    return host
