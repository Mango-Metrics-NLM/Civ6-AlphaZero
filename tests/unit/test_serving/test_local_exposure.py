"""LF-G14: web interface and inference server default to loopback only."""

from __future__ import annotations

import logging

from alpha_zero.web.config import WebConfig


class TestWebConfigLoopbackDefaults:
    def test_default_host_is_loopback(self):
        assert WebConfig().host == "127.0.0.1"

    def test_from_env_without_base_is_loopback(self, monkeypatch):
        monkeypatch.delenv("CIV6AZ_WEB_HOST", raising=False)
        assert WebConfig.from_env().host == "127.0.0.1"

    def test_default_cors_is_loopback_only_without_wildcard(self):
        origins = WebConfig().cors_origins
        assert "*" not in origins
        assert "http://127.0.0.1:8080" in origins
        assert "http://localhost:8080" in origins
        assert "http://[::1]:8080" in origins
        for origin in origins:
            assert origin.split("://", 1)[1].rsplit(":", 1)[0] in {
                "127.0.0.1",
                "localhost",
                "[::1]",
            }

    def test_default_cors_follows_port(self):
        origins = WebConfig(port=9123).cors_origins
        assert "http://127.0.0.1:9123" in origins
        assert not any(":8080" in o for o in origins)

    def test_explicit_cors_is_respected(self):
        assert WebConfig(cors_origins=["https://example.com"]).cors_origins == [
            "https://example.com"
        ]

    def test_loopback_host_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="alpha_zero.web.config"):
            WebConfig(host="127.0.0.1")
            WebConfig(host="localhost")
            WebConfig(host="::1")
        assert not [r for r in caplog.records if "non-loopback" in r.getMessage()]

    def test_explicit_all_interfaces_works_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="alpha_zero.web.config"):
            config = WebConfig(host="0.0.0.0")
        assert config.host == "0.0.0.0"
        assert any("non-loopback" in r.getMessage() for r in caplog.records)
