"""
Unit tests for InferenceEngine.

Tests model loading, caching, and inference logic.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from alpha_zero.cloud.gcp.storage import ModelMetadata
from alpha_zero.inference.api_schemas import GameState, InferenceRequest, MCTSType
from alpha_zero.inference.inference_engine import (
    InferenceConfig,
    InferenceEngine,
    ModelCache,
)


@pytest.fixture
def mock_storage():
    """Create mock GCS storage."""
    storage = Mock()

    # Mock model state dict
    state_dict = {
        "encoder.conv1.weight": torch.randn(64, 13, 3, 3),
        "policy_head.fc.weight": torch.randn(376, 64),
        "value_head.fc.weight": torch.randn(1, 64),
    }

    # Mock metadata
    metadata = ModelMetadata(version=100, iteration=50)

    storage.load_model.return_value = (state_dict, metadata)
    storage.load_champion.return_value = (state_dict, metadata)

    return storage


@pytest.fixture
def inference_config():
    """Create test inference config."""
    return InferenceConfig(
        device="cpu",
        model_cache_size=2,
        batch_timeout_ms=10.0,
    )


@pytest.fixture
def game_state():
    """Create test game state."""
    spatial = np.random.randn(13, 8, 8).flatten().tolist()
    vector = np.random.randn(64).tolist()
    action_mask = [1] * 376

    return GameState(
        spatial=spatial,
        vector=vector,
        action_mask=action_mask,
        board_size=8,
        turn_number=5,
    )


class TestModelCache:
    """Tests for ModelCache."""

    def test_cache_get_miss(self):
        """Test cache miss returns None."""
        cache = ModelCache(max_size=3, device="cpu")
        model = cache.get(version=1)
        assert model is None

    def test_cache_put_and_get(self):
        """Test caching and retrieving model."""
        cache = ModelCache(max_size=3, device="cpu")

        # Create mock model
        model = Mock()

        # Put in cache
        cache.put(version=1, model=model)

        # Get from cache
        retrieved = cache.get(version=1)
        assert retrieved is model

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ModelCache(max_size=2, device="cpu")

        # Add 3 models (should evict first)
        model1 = Mock()
        model2 = Mock()
        model3 = Mock()

        cache.put(version=1, model=model1)
        cache.put(version=2, model=model2)
        cache.put(version=3, model=model3)  # Evicts v1

        # v1 should be evicted
        assert cache.get(version=1) is None
        assert cache.get(version=2) is model2
        assert cache.get(version=3) is model3

    def test_cache_access_updates_order(self):
        """Test that accessing model updates LRU order."""
        cache = ModelCache(max_size=2, device="cpu")

        model1 = Mock()
        model2 = Mock()
        model3 = Mock()

        cache.put(version=1, model=model1)
        cache.put(version=2, model=model2)

        # Access v1 (makes it most recent)
        cache.get(version=1)

        # Add v3 (should evict v2, not v1)
        cache.put(version=3, model=model3)

        assert cache.get(version=1) is model1
        assert cache.get(version=2) is None  # Evicted
        assert cache.get(version=3) is model3

    def test_cache_clear(self):
        """Test clearing all cached models."""
        cache = ModelCache(max_size=3, device="cpu")

        cache.put(version=1, model=Mock())
        cache.put(version=2, model=Mock())

        cache.clear()

        assert cache.get(version=1) is None
        assert cache.get(version=2) is None
        assert len(cache.cache) == 0


class TestInferenceEngine:
    """Tests for InferenceEngine."""

    @pytest.mark.asyncio
    async def test_load_model_latest(self, mock_storage, inference_config):
        """Test loading latest champion model."""
        engine = InferenceEngine(mock_storage, inference_config)

        with patch("alpha_zero.inference.inference_engine.Civ6TransformerNetwork"):
            version = await engine.load_model()

        assert version == 100
        assert engine.current_version == 100
        mock_storage.load_champion.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_specific_version(self, mock_storage, inference_config):
        """Test loading specific model version."""
        engine = InferenceEngine(mock_storage, inference_config)

        with patch("alpha_zero.inference.inference_engine.Civ6TransformerNetwork"):
            version = await engine.load_model(version=50)

        assert version == 50
        assert engine.current_version == 50
        mock_storage.load_model.assert_called_once_with(50)

    @pytest.mark.asyncio
    async def test_load_model_from_cache(self, mock_storage, inference_config):
        """Test loading cached model (no GCS call)."""
        engine = InferenceEngine(mock_storage, inference_config)

        with patch("alpha_zero.inference.inference_engine.Civ6TransformerNetwork"):
            # First load
            await engine.load_model(version=100)
            mock_storage.load_model.reset_mock()

            # Second load (from cache)
            version = await engine.load_model(version=100)

        assert version == 100
        mock_storage.load_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_model_force_reload(self, mock_storage, inference_config):
        """Test force reloading model bypasses cache."""
        engine = InferenceEngine(mock_storage, inference_config)

        with patch("alpha_zero.inference.inference_engine.Civ6TransformerNetwork"):
            # First load
            await engine.load_model(version=100)
            mock_storage.load_model.reset_mock()

            # Force reload
            version = await engine.load_model(version=100, force=True)

        assert version == 100
        mock_storage.load_model.assert_called_once_with(100)

    @pytest.mark.asyncio
    async def test_prepare_inputs(self, mock_storage, inference_config, game_state):
        """Test game state to tensor conversion."""
        engine = InferenceEngine(mock_storage, inference_config)

        spatial, vector, action_mask = engine._prepare_inputs(game_state)

        # Check shapes
        assert spatial.shape == (1, 13, 8, 8)
        assert vector.shape == (1, 64)
        assert action_mask.shape == (376,)

        # Check device
        assert spatial.device.type == "cpu"
        assert vector.device.type == "cpu"

    def test_get_health_no_model(self, mock_storage, inference_config):
        """Test health check with no model loaded."""
        engine = InferenceEngine(mock_storage, inference_config)

        health = engine.get_health()

        assert health["status"] == "degraded"
        assert health["model_loaded"] is False
        assert health["model_version"] is None

    def test_get_health_with_model(self, mock_storage, inference_config):
        """Test health check with loaded model."""
        engine = InferenceEngine(mock_storage, inference_config)
        engine.current_model = Mock()
        engine.current_version = 100

        health = engine.get_health()

        assert health["status"] == "healthy"
        assert health["model_loaded"] is True
        assert health["model_version"] == 100

    def test_get_metrics_initial(self, mock_storage, inference_config):
        """Test initial metrics before any requests."""
        engine = InferenceEngine(mock_storage, inference_config)

        metrics = engine.get_metrics()

        assert metrics["total_requests"] == 0
        assert metrics["requests_per_second"] == 0.0
        assert metrics["average_latency_ms"] == 0.0

    def test_get_metrics_after_requests(self, mock_storage, inference_config):
        """Test metrics after processing requests."""
        engine = InferenceEngine(mock_storage, inference_config)

        # Simulate requests
        engine.total_requests = 100
        engine.latency_history.extend([100.0, 200.0, 150.0])

        metrics = engine.get_metrics()

        assert metrics["total_requests"] == 100
        assert metrics["average_latency_ms"] == 150.0
        assert metrics["p95_latency_ms"] > 0


@pytest.mark.integration
class TestInferenceEngineIntegration:
    """Integration tests with real network."""

    @pytest.mark.asyncio
    async def test_end_to_end_inference(self, mock_storage, inference_config, game_state):
        """Test full inference pipeline (mocked MCTS)."""
        engine = InferenceEngine(mock_storage, inference_config)

        # Mock network forward pass
        with patch("alpha_zero.inference.inference_engine.Civ6TransformerNetwork") as MockNet:
            mock_model = Mock()
            mock_model.return_value = (
                torch.randn(1, 376),  # policy_logits
                torch.tensor([[0.5]]),  # value
            )
            MockNet.return_value = mock_model

            await engine.load_model()

            # Create request
            request = InferenceRequest(
                game_state=game_state,
                mcts_simulations=16,
                mcts_type=MCTSType.GUMBEL,
                temperature=1.0,
            )

            # Run inference
            with patch("alpha_zero.inference.inference_engine.GumbelMCTS"):
                action_dist = await engine.infer(request)

            # Verify output
            assert len(action_dist.action_probs) == 376
            assert 0 <= action_dist.action_id < 376
            assert -1.0 <= action_dist.value_estimate <= 1.0
            assert len(action_dist.top_k_actions) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
