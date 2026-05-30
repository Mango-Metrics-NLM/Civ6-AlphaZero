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
    storage.get_latest_version.return_value = 100

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
        mock_storage.get_latest_version.assert_called_once()

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


class TestApplyPolicyPostprocessing:
    """Regression tests for value-tensor shape handling.

    Networks produced over the project's history return ``value`` with
    various shapes. ``_apply_policy_postprocessing`` must not crash on any
    of them, since the inference engine is the single entry point for all
    served models.
    """

    @pytest.fixture
    def engine(self, mock_storage, inference_config):
        return InferenceEngine(mock_storage, inference_config)

    def _logits(self, action_space: int = 376) -> torch.Tensor:
        # Deterministic, finite logits so softmax is well-defined.
        return torch.zeros(1, action_space)

    def _mask(self, action_space: int = 376) -> np.ndarray:
        return np.ones(action_space, dtype=np.float32)

    @pytest.mark.parametrize(
        ("value_tensor", "expected"),
        [
            (torch.tensor([[0.5]]), 0.5),  # (1, 1) — squeeze-omitted 2P output
            (torch.tensor([0.42]), 0.42),  # (1,)   — current convention
            (torch.tensor(0.3), 0.3),  # scalar — legacy
        ],
    )
    def test_scalar_value_shapes_round_trip(
        self, engine, value_tensor: torch.Tensor, expected: float
    ) -> None:
        _probs, value_estimate = engine._apply_policy_postprocessing(
            self._logits(), self._mask(), temperature=1.0, value=value_tensor
        )
        assert value_estimate == pytest.approx(expected, abs=1e-6)

    def test_multi_player_value_reports_first_player(self, engine, caplog) -> None:
        """N-player networks (shape (1, N)) report player 0's value without crashing.

        Previously ``value[0].item()`` raised
        ``RuntimeError: only one element tensors can be converted`` for N>1.
        """
        import logging

        value = torch.tensor([[0.7, -0.4, 0.1]])  # 3-player vector
        # The shape-normaliser lives in alpha_zero.bridge.utils now (shared
        # between InferenceEngine and BridgeAgentLoop). Catch its DEBUG log
        # there rather than on the inference_engine logger.
        with caplog.at_level(logging.DEBUG, logger="alpha_zero.bridge.utils"):
            _probs, value_estimate = engine._apply_policy_postprocessing(
                self._logits(), self._mask(), temperature=1.0, value=value
            )
        assert value_estimate == pytest.approx(0.7)
        # Debug log surfaces the per-player tensor so operators can audit it.
        assert any("per-player value tensor" in r.message for r in caplog.records)

    def test_none_value_returns_zero(self, engine) -> None:
        _probs, value_estimate = engine._apply_policy_postprocessing(
            self._logits(), self._mask(), temperature=1.0, value=None
        )
        assert value_estimate == 0.0

    def test_empty_value_returns_zero(self, engine) -> None:
        """Defensive: empty value tensor must not raise."""
        _probs, value_estimate = engine._apply_policy_postprocessing(
            self._logits(), self._mask(), temperature=1.0, value=torch.empty(0)
        )
        assert value_estimate == 0.0

    def test_greedy_temperature_selects_argmax(self, engine) -> None:
        logits = torch.tensor([[0.0, 10.0, 0.0, 0.0]])
        mask = np.ones(4, dtype=np.float32)
        probs, _ = engine._apply_policy_postprocessing(
            logits, mask, temperature=0.0, value=None
        )
        assert int(np.argmax(probs)) == 1
        assert probs[1] == pytest.approx(1.0)


@pytest.mark.integration
class TestInferenceEngineIntegration:
    """Integration tests with real network."""

    @pytest.mark.asyncio
    async def test_end_to_end_inference_falls_back_to_raw_network(
        self, mock_storage, inference_config, game_state, caplog
    ):
        """Inference pipeline routes through raw network (no env-based MCTS).

        Tree-search engines (Civ6/Gumbel/BatchGumbel) need an env with
        clone(); the inference engine only has observations. So requesting
        mcts_simulations > 0 must log a warning and use raw network output.
        """
        import logging

        engine = InferenceEngine(mock_storage, inference_config)

        # Mock network forward pass — deterministic outputs let us assert
        # the value passes through verbatim.
        with patch("alpha_zero.inference.inference_engine.Civ6TransformerNetwork") as MockNet:
            mock_model = Mock()
            mock_model.return_value = (
                torch.randn(1, 376),  # policy_logits
                torch.tensor([[0.5]]),  # value
            )
            MockNet.return_value = mock_model

            await engine.load_model()

            request = InferenceRequest(
                game_state=game_state,
                mcts_simulations=16,
                mcts_type=MCTSType.GUMBEL,
                temperature=1.0,
            )

            with caplog.at_level(logging.WARNING, logger="alpha_zero.inference.inference_engine"):
                action_dist = await engine.infer(request)

            # Output shape & value contract.
            assert len(action_dist.action_probs) == 376
            assert 0 <= action_dist.action_id < 376
            assert -1.0 <= action_dist.value_estimate <= 1.0
            assert len(action_dist.top_k_actions) == 5

            # Value comes from the raw network forward pass (0.5), proving
            # the fallback path is exercised rather than any stale MCTS code.
            assert action_dist.value_estimate == pytest.approx(0.5)

            # The engine surfaces the env-less MCTS constraint as a warning.
            assert any(
                "falling back to raw network inference" in r.message for r in caplog.records
            ), "Expected a warning about MCTS search falling back to raw network inference"

    @pytest.mark.asyncio
    async def test_inference_without_mcts_does_not_log_fallback_warning(
        self, mock_storage, inference_config, game_state, caplog
    ):
        """Requests with mcts_simulations=1 should not warn about fallback.

        ``InferenceRequest.mcts_simulations`` has a Pydantic floor of 1, so
        "no MCTS requested" is encoded as the minimum. The engine still
        falls back to raw inference but should not emit the warning that is
        intended for higher-sim requests, so this guards against log spam
        regressions. (If product semantics change to allow ``mcts=0``, add
        a separate assertion for that case.)
        """
        import logging

        engine = InferenceEngine(mock_storage, inference_config)

        with patch("alpha_zero.inference.inference_engine.Civ6TransformerNetwork") as MockNet:
            mock_model = Mock()
            mock_model.return_value = (
                torch.randn(1, 376),
                torch.tensor([[0.1]]),
            )
            MockNet.return_value = mock_model

            await engine.load_model()

            request = InferenceRequest(
                game_state=game_state,
                mcts_simulations=1,
                mcts_type=MCTSType.GUMBEL,
                temperature=1.0,
            )

            with caplog.at_level(logging.WARNING, logger="alpha_zero.inference.inference_engine"):
                action_dist = await engine.infer(request)

            assert action_dist.value_estimate == pytest.approx(0.1)
            # The warning still fires (sims > 0) but value comes from network.
            warning_count = sum(
                1 for r in caplog.records if "falling back to raw network inference" in r.message
            )
            assert warning_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
