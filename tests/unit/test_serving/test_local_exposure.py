"""LF-G14: web interface and inference server default to loopback only."""

from __future__ import annotations

import logging
from pathlib import Path

from alpha_zero.utils.network import resolve_bind_host
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


class TestProductionPreset:
    def test_production_preset_is_loopback_with_loopback_cors(self):
        config = WebConfig.from_preset("production")
        assert config.host == "127.0.0.1"
        assert "*" not in config.cors_origins


class TestResolveBindHost:
    """The inference server's bind address (no fastapi needed to test it)."""

    def test_default_is_loopback_and_silent(self, monkeypatch, caplog):
        monkeypatch.delenv("INFERENCE_HOST", raising=False)
        with caplog.at_level(logging.WARNING):
            assert resolve_bind_host("INFERENCE_HOST", logging.getLogger("t"), "x") == "127.0.0.1"
        assert not caplog.records

    def test_blank_env_falls_back_to_loopback(self, monkeypatch):
        monkeypatch.setenv("INFERENCE_HOST", "  ")
        assert resolve_bind_host("INFERENCE_HOST", logging.getLogger("t"), "x") == "127.0.0.1"

    def test_all_interfaces_honoured_and_warned(self, monkeypatch, caplog):
        monkeypatch.setenv("INFERENCE_HOST", "0.0.0.0")
        with caplog.at_level(logging.WARNING):
            host = resolve_bind_host("INFERENCE_HOST", logging.getLogger("t"), "x")
        assert host == "0.0.0.0"
        assert any("non-loopback" in r.getMessage() for r in caplog.records)


def test_inference_dockerfile_binds_all_interfaces_explicitly():
    """Without this ENV the container would bind loopback and be unreachable."""
    text = (Path(__file__).parents[3] / "dockerfiles" / "Dockerfile.inference").read_text()
    assert "ENV INFERENCE_HOST=0.0.0.0" in text
