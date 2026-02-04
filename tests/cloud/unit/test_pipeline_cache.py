"""Unit tests for pipeline_cache module."""
from __future__ import annotations
import os
import pytest
from alpha_zero.cloud.gcp.pipeline_cache import (
    CacheEntry, ComponentSignature, ComponentType,
    PipelineCache, PipelineCacheConfig, get_pipeline_cache,
)

class TestComponentType:
    def test_component_type_values(self) -> None:
        assert ComponentType.SELF_PLAY.value == "self_play"
        assert ComponentType.TRAINING.value == "training"
        assert ComponentType.EVALUATION.value == "evaluation"

class TestComponentSignature:
    def test_creation(self) -> None:
        sig = ComponentSignature(
            component_type=ComponentType.SELF_PLAY,
            model_version=1,
            config_hash="abc123",
            code_hash="def456",
        )
        assert sig.component_type == ComponentType.SELF_PLAY
        assert sig.model_version == 1

    def test_to_cache_key(self) -> None:
        sig = ComponentSignature(
            component_type=ComponentType.TRAINING,
            model_version=5,
            config_hash="abc",
            code_hash="def",
        )
        key = sig.to_cache_key()
        assert "training" in key
        assert "5" in key

    def test_to_dict(self) -> None:
        sig = ComponentSignature(
            component_type=ComponentType.EVALUATION,
            model_version=10,
            config_hash="hash1",
            code_hash="hash2",
        )
        data = sig.to_dict()
        assert data["component_type"] == "evaluation"
        assert data["model_version"] == 10

class TestCacheEntry:
    def test_creation(self) -> None:
        sig = ComponentSignature(
            component_type=ComponentType.SELF_PLAY,
            model_version=1,
            config_hash="hash1",
        )
        entry = CacheEntry(
            signature=sig,
            output_path="gs://bucket/output",
            created_at="2024-01-01T00:00:00Z",
        )
        assert entry.signature == sig
        assert entry.output_path == "gs://bucket/output"

    def test_is_expired_not_expired(self) -> None:
        sig = ComponentSignature(
            component_type=ComponentType.SELF_PLAY,
            model_version=1,
            config_hash="hash1",
        )
        entry = CacheEntry(
            signature=sig,
            output_path="gs://test",
            created_at="2024-01-01T00:00:00Z",
            valid_until=None,  # No expiration
        )
        assert entry.is_expired() is False

class TestPipelineCacheConfig:
    def test_default_values(self) -> None:
        config = PipelineCacheConfig()
        assert config.enabled is True
        assert config.ttl_hours == 168
        assert config.max_entries == 1000

    def test_from_env(self) -> None:
        os.environ["CIV6AZ_CACHE_ENABLED"] = "true"
        os.environ["CIV6AZ_CACHE_TTL_HOURS"] = "72"
        try:
            config = PipelineCacheConfig.from_env()
            assert config.enabled is True
            assert config.ttl_hours == 72
        finally:
            del os.environ["CIV6AZ_CACHE_ENABLED"]
            del os.environ["CIV6AZ_CACHE_TTL_HOURS"]

class TestPipelineCache:
    def test_init(self) -> None:
        config = PipelineCacheConfig()
        config.enabled = False
        cache = PipelineCache(config=config, bucket="test-bucket")
        assert cache._bucket == "test-bucket"
        assert cache.config.enabled is False

    def test_compute_signature(self) -> None:
        config = PipelineCacheConfig()
        cache = PipelineCache(config=config, bucket="test-bucket")
        sig = cache.compute_signature(
            component_type=ComponentType.SELF_PLAY,
            model_version=1,
            config={"key": "value"},
        )
        assert isinstance(sig, ComponentSignature)
        assert sig.component_type == ComponentType.SELF_PLAY

    def test_get_statistics(self) -> None:
        config = PipelineCacheConfig()
        cache = PipelineCache(config=config, bucket="test-bucket")
        stats = cache.get_statistics()
        assert "enabled" in stats
        assert "ttl_hours" in stats

class TestGetPipelineCache:
    def test_returns_cache(self) -> None:
        cache = get_pipeline_cache(bucket="test-bucket")
        assert isinstance(cache, PipelineCache)