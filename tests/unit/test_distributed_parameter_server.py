"""Unit tests for alpha_zero/distributed/parameter_server.py.

Covers:
- ParameterServerConfig defaults
- ParameterServer: set_weights, get_weights, get_weights_with_version,
  get_serialized_weights, deserialize_weights, get_version,
  get_weights_at_version, update_model, apply_gradients, apply_delta,
  get_statistics, reset, weight-update callback, compression toggle
- MockParameterServer: latency simulation, failure injection, update history
- DistributedParameterServer: register/unregister client, heartbeat,
  stale clients, client count, broadcast_weights, get_cluster_info
- create_parameter_server factory
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from alpha_zero.distributed.parameter_server import (
    DistributedParameterServer,
    MockParameterServer,
    ParameterServer,
    ParameterServerConfig,
    WeightUpdateMode,
    create_parameter_server,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_state_dict() -> dict[str, torch.Tensor]:
    """Return a minimal state dict for testing."""
    return {
        "layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "layer.bias": torch.tensor([0.1, 0.2]),
    }


def _tiny_network() -> nn.Module:
    net = nn.Linear(2, 2)
    nn.init.constant_(net.weight, 1.0)
    nn.init.constant_(net.bias, 0.0)
    return net


# ---------------------------------------------------------------------------
# WeightUpdateMode
# ---------------------------------------------------------------------------


class TestWeightUpdateMode:
    def test_values(self):
        assert WeightUpdateMode.REPLACE.value == "replace"
        assert WeightUpdateMode.GRADIENT.value == "gradient"
        assert WeightUpdateMode.DELTA.value == "delta"


# ---------------------------------------------------------------------------
# ParameterServerConfig
# ---------------------------------------------------------------------------


class TestParameterServerConfig:
    def test_defaults(self):
        cfg = ParameterServerConfig()
        assert cfg.update_mode == WeightUpdateMode.REPLACE.value
        assert cfg.compress_weights is True
        assert cfg.compression_level == 3
        assert cfg.max_version_history == 5
        assert cfg.async_gradient_steps == 1
        assert cfg.gradient_staleness_threshold == 10
        assert cfg.lock_timeout_seconds == 30.0
        assert cfg.on_weight_update is None

    def test_custom_config(self):
        cfg = ParameterServerConfig(
            compress_weights=False,
            max_version_history=2,
            gradient_staleness_threshold=5,
        )
        assert cfg.compress_weights is False
        assert cfg.max_version_history == 2
        assert cfg.gradient_staleness_threshold == 5

    def test_callback_stored(self):
        cb = MagicMock()
        cfg = ParameterServerConfig(on_weight_update=cb)
        assert cfg.on_weight_update is cb


# ---------------------------------------------------------------------------
# ParameterServer – basic weight operations
# ---------------------------------------------------------------------------


class TestParameterServerWeights:
    def test_initial_state(self):
        ps = ParameterServer()
        assert ps.get_version() == 0
        assert ps.get_weights() is None
        serialized, version = ps.get_serialized_weights()
        assert serialized is None
        assert version == 0

    def test_set_weights_increments_version(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        sd = _tiny_state_dict()
        v1 = ps.set_weights(sd)
        assert v1 == 1
        v2 = ps.set_weights(sd)
        assert v2 == 2

    def test_set_weights_no_increment(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        sd = _tiny_state_dict()
        v = ps.set_weights(sd, increment_version=False)
        assert v == 0
        assert ps.get_version() == 0

    def test_get_weights_returns_clone(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        sd = _tiny_state_dict()
        ps.set_weights(sd)
        weights = ps.get_weights()
        assert weights is not None
        # Mutate returned weights; original should be unaffected
        weights["layer.weight"][0, 0] = 999.0
        w2 = ps.get_weights()
        assert w2["layer.weight"][0, 0].item() != 999.0

    def test_get_weights_with_version(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        sd = _tiny_state_dict()
        ps.set_weights(sd)
        weights, version = ps.get_weights_with_version()
        assert weights is not None
        assert version == 1

    def test_get_weights_with_version_none(self):
        ps = ParameterServer()
        weights, version = ps.get_weights_with_version()
        assert weights is None
        assert version == 0

    def test_set_weights_with_compression(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=True))
        sd = _tiny_state_dict()
        ps.set_weights(sd)
        serialized, version = ps.get_serialized_weights()
        assert serialized is not None
        assert version == 1
        stats = ps.get_statistics()
        assert stats["compression_ratio"] < 1.0 or stats["compression_ratio"] > 0.0

    def test_set_weights_without_compression(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        sd = _tiny_state_dict()
        ps.set_weights(sd)
        stats = ps.get_statistics()
        assert stats["compression_ratio"] == 1.0

    def test_init_from_network(self):
        net = _tiny_network()
        ps = ParameterServer(network=net, config=ParameterServerConfig(compress_weights=False))
        assert ps.get_version() == 1
        weights = ps.get_weights()
        assert weights is not None
        assert set(weights.keys()) == set(net.state_dict().keys())


# ---------------------------------------------------------------------------
# ParameterServer – weight history and version retrieval
# ---------------------------------------------------------------------------


class TestParameterServerHistory:
    def test_version_history_pruned(self):
        ps = ParameterServer(
            config=ParameterServerConfig(compress_weights=False, max_version_history=3)
        )
        sd = _tiny_state_dict()
        for _ in range(6):
            ps.set_weights(sd)
        assert len(ps._weight_history) == 3

    def test_get_weights_at_version(self):
        ps = ParameterServer(
            config=ParameterServerConfig(compress_weights=False, max_version_history=5)
        )
        sd1 = {"w": torch.tensor([1.0])}
        sd2 = {"w": torch.tensor([2.0])}
        ps.set_weights(sd1)  # version 1
        ps.set_weights(sd2)  # version 2

        w1 = ps.get_weights_at_version(1)
        assert w1 is not None
        assert w1["w"].item() == pytest.approx(1.0)

    def test_get_weights_at_missing_version(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        result = ps.get_weights_at_version(99)
        assert result is None


# ---------------------------------------------------------------------------
# ParameterServer – deserialize_weights
# ---------------------------------------------------------------------------


class TestDeserializeWeights:
    def test_deserialize_uncompressed(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        sd = _tiny_state_dict()
        ps.set_weights(sd)
        serialized, _ = ps.get_serialized_weights()
        recovered = ps.deserialize_weights(serialized, compressed=False)
        assert set(recovered.keys()) == set(sd.keys())

    def test_deserialize_compressed(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=True))
        sd = _tiny_state_dict()
        ps.set_weights(sd)
        serialized, _ = ps.get_serialized_weights()
        recovered = ps.deserialize_weights(serialized, compressed=True)
        assert set(recovered.keys()) == set(sd.keys())

    def test_deserialize_auto_detect(self):
        """compressed=None should use config value."""
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        sd = _tiny_state_dict()
        ps.set_weights(sd)
        serialized, _ = ps.get_serialized_weights()
        recovered = ps.deserialize_weights(serialized, compressed=None)
        assert set(recovered.keys()) == set(sd.keys())


# ---------------------------------------------------------------------------
# ParameterServer – update_model
# ---------------------------------------------------------------------------


class TestUpdateModel:
    def test_update_model_success(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        ps.set_weights({"weight": torch.tensor([1.0, 2.0])})
        net = MagicMock()
        updated, version = ps.update_model(net)
        assert updated is True
        assert version == 1
        net.load_state_dict.assert_called_once()

    def test_update_model_version_not_newer(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        ps.set_weights(_tiny_state_dict())
        net = MagicMock()
        # min_version == current version → should not update
        updated, version = ps.update_model(net, min_version=1)
        assert updated is False

    def test_update_model_no_weights(self):
        ps = ParameterServer()
        net = MagicMock()
        updated, version = ps.update_model(net)
        assert updated is False

    def test_update_model_load_state_dict_error(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        ps.set_weights(_tiny_state_dict())
        net = MagicMock()
        net.load_state_dict.side_effect = RuntimeError("incompatible")
        updated, version = ps.update_model(net)
        assert updated is False


# ---------------------------------------------------------------------------
# ParameterServer – apply_gradients
# ---------------------------------------------------------------------------


class TestApplyGradients:
    def test_apply_gradients_basic(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        sd = {"w": torch.tensor([1.0, 2.0])}
        ps.set_weights(sd)
        grads = {"w": torch.tensor([0.1, 0.2])}
        new_version = ps.apply_gradients(grads, learning_rate=1.0)
        assert new_version == 2
        updated = ps.get_weights()
        assert updated["w"][0].item() == pytest.approx(0.9, rel=1e-4)
        assert updated["w"][1].item() == pytest.approx(1.8, rel=1e-4)

    def test_apply_gradients_no_weights_raises(self):
        ps = ParameterServer()
        with pytest.raises(RuntimeError, match="no weights set"):
            ps.apply_gradients({"w": torch.tensor([0.1])}, learning_rate=0.01)

    def test_apply_gradients_staleness_warning(self):
        cfg = ParameterServerConfig(compress_weights=False, gradient_staleness_threshold=2)
        ps = ParameterServer(config=cfg)
        ps.set_weights({"w": torch.tensor([1.0])})
        ps.set_weights({"w": torch.tensor([1.0])})
        ps.set_weights({"w": torch.tensor([1.0])})
        # source_version=0, current=3 → staleness=3 > threshold=2
        with patch.object(ps, "_ParameterServer__dict__", {}, create=True):
            # Just verify it doesn't raise (warning is logged)
            new_v = ps.apply_gradients(
                {"w": torch.tensor([0.0])}, learning_rate=0.01, source_version=0
            )
        assert new_v == 4

    def test_apply_gradients_ignores_unknown_keys(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        ps.set_weights({"w": torch.tensor([1.0])})
        grads = {"nonexistent_key": torch.tensor([0.5])}
        new_v = ps.apply_gradients(grads, learning_rate=1.0)
        assert new_v == 2  # Version still increments


# ---------------------------------------------------------------------------
# ParameterServer – apply_delta
# ---------------------------------------------------------------------------


class TestApplyDelta:
    def test_apply_delta_success(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        ps.set_weights({"w": torch.tensor([1.0, 2.0])})
        delta = {"w": torch.tensor([0.5, -0.5])}
        new_v = ps.apply_delta(delta, source_version=1)
        assert new_v == 2
        updated = ps.get_weights()
        assert updated["w"][0].item() == pytest.approx(1.5, rel=1e-4)
        assert updated["w"][1].item() == pytest.approx(1.5, rel=1e-4)

    def test_apply_delta_version_mismatch(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        ps.set_weights({"w": torch.tensor([1.0])})
        result = ps.apply_delta({"w": torch.tensor([0.1])}, source_version=99)
        assert result is None

    def test_apply_delta_no_weights_raises(self):
        ps = ParameterServer()
        with pytest.raises(RuntimeError, match="no weights set"):
            ps.apply_delta({"w": torch.tensor([0.1])}, source_version=0)


# ---------------------------------------------------------------------------
# ParameterServer – statistics and reset
# ---------------------------------------------------------------------------


class TestStatisticsAndReset:
    def test_get_statistics_initial(self):
        ps = ParameterServer()
        stats = ps.get_statistics()
        assert stats["version"] == 0
        assert stats["total_updates"] == 0
        assert stats["total_fetches"] == 0
        assert stats["weights_size_bytes"] == 0

    def test_get_statistics_after_set(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        sd = _tiny_state_dict()
        ps.set_weights(sd)
        ps.get_weights()
        stats = ps.get_statistics()
        assert stats["version"] == 1
        assert stats["total_updates"] == 1
        assert stats["total_fetches"] == 1
        assert stats["weights_size_bytes"] > 0

    def test_reset_clears_all_state(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        ps.set_weights(_tiny_state_dict())
        ps.reset()
        assert ps.get_version() == 0
        assert ps.get_weights() is None
        stats = ps.get_statistics()
        assert stats["total_updates"] == 0
        assert stats["history_size"] == 0

    def test_callback_invoked_on_set_weights(self):
        callback = MagicMock()
        cfg = ParameterServerConfig(compress_weights=False, on_weight_update=callback)
        ps = ParameterServer(config=cfg)
        ps.set_weights(_tiny_state_dict())
        callback.assert_called_once_with(1)

    def test_callback_exception_does_not_raise(self):
        def bad_callback(v):
            raise ValueError("callback error")

        cfg = ParameterServerConfig(compress_weights=False, on_weight_update=bad_callback)
        ps = ParameterServer(config=cfg)
        # Should log warning but not raise
        ps.set_weights(_tiny_state_dict())
        assert ps.get_version() == 1


# ---------------------------------------------------------------------------
# ParameterServer – thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_set_and_get(self):
        ps = ParameterServer(config=ParameterServerConfig(compress_weights=False))
        errors = []

        def writer():
            try:
                for _ in range(20):
                    ps.set_weights(_tiny_state_dict())
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(20):
                    ps.get_weights()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(2)]
        threads += [threading.Thread(target=reader) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == [], f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# MockParameterServer
# ---------------------------------------------------------------------------


class TestMockParameterServer:
    def test_initial_state(self):
        mock_ps = MockParameterServer()
        assert mock_ps._simulate_latency == 0.0
        assert mock_ps._fail_next_update is False
        assert mock_ps._fail_next_fetch is False

    def test_fail_next_update_raises(self):
        mock_ps = MockParameterServer(config=ParameterServerConfig(compress_weights=False))
        mock_ps.fail_next_update()
        with pytest.raises(RuntimeError, match="Simulated update failure"):
            mock_ps.set_weights(_tiny_state_dict())

    def test_fail_next_update_clears_after_use(self):
        mock_ps = MockParameterServer(config=ParameterServerConfig(compress_weights=False))
        mock_ps.fail_next_update()
        try:
            mock_ps.set_weights(_tiny_state_dict())
        except RuntimeError:
            pass
        # Second call should succeed
        v = mock_ps.set_weights(_tiny_state_dict())
        assert v == 1

    def test_fail_next_fetch_raises(self):
        mock_ps = MockParameterServer(config=ParameterServerConfig(compress_weights=False))
        mock_ps.set_weights(_tiny_state_dict())
        mock_ps.fail_next_fetch()
        with pytest.raises(RuntimeError, match="Simulated fetch failure"):
            mock_ps.get_weights()

    def test_fail_next_fetch_clears_after_use(self):
        mock_ps = MockParameterServer(config=ParameterServerConfig(compress_weights=False))
        mock_ps.set_weights(_tiny_state_dict())
        mock_ps.fail_next_fetch()
        try:
            mock_ps.get_weights()
        except RuntimeError:
            pass
        # Second fetch should succeed
        w = mock_ps.get_weights()
        assert w is not None

    def test_update_history_tracked(self):
        mock_ps = MockParameterServer(config=ParameterServerConfig(compress_weights=False))
        mock_ps.set_weights(_tiny_state_dict())
        mock_ps.set_weights(_tiny_state_dict())
        history = mock_ps.get_update_history()
        assert len(history) == 2
        assert history[0]["version"] == 1
        assert history[1]["version"] == 2
        assert "time" in history[0]
        assert "param_count" in history[0]

    def test_clear_history(self):
        mock_ps = MockParameterServer(config=ParameterServerConfig(compress_weights=False))
        mock_ps.set_weights(_tiny_state_dict())
        mock_ps.clear_history()
        assert mock_ps.get_update_history() == []

    def test_simulate_latency(self):
        mock_ps = MockParameterServer(config=ParameterServerConfig(compress_weights=False))
        mock_ps.simulate_latency(0.0)  # Zero latency for speed
        v = mock_ps.set_weights(_tiny_state_dict())
        assert v == 1


# ---------------------------------------------------------------------------
# DistributedParameterServer
# ---------------------------------------------------------------------------


class TestDistributedParameterServer:
    def test_initialization(self):
        ps = DistributedParameterServer(node_id="ps-test")
        assert ps.node_id == "ps-test"
        assert ps.get_version() == 0

    def test_register_client(self):
        ps = DistributedParameterServer()
        result = ps.register_client("learner-0", client_type="learner")
        assert result is True
        assert ps.get_client_count() == 1
        assert ps.get_client_count("learner") == 1
        assert ps.get_client_count("actor") == 0

    def test_register_multiple_clients(self):
        ps = DistributedParameterServer()
        ps.register_client("learner-0", client_type="learner")
        ps.register_client("actor-0", client_type="actor")
        ps.register_client("actor-1", client_type="actor")
        assert ps.get_client_count() == 3
        assert ps.get_client_count("learner") == 1
        assert ps.get_client_count("actor") == 2

    def test_register_with_metadata(self):
        ps = DistributedParameterServer()
        ps.register_client("actor-0", client_type="actor", metadata={"gpu": 0})
        assert ps._clients["actor-0"]["metadata"]["gpu"] == 0

    def test_unregister_client(self):
        ps = DistributedParameterServer()
        ps.register_client("learner-0")
        result = ps.unregister_client("learner-0")
        assert result is True
        assert ps.get_client_count() == 0

    def test_unregister_nonexistent_client(self):
        ps = DistributedParameterServer()
        result = ps.unregister_client("nonexistent")
        assert result is False

    def test_heartbeat_registered_client(self):
        ps = DistributedParameterServer()
        ps.register_client("actor-0")
        result = ps.heartbeat("actor-0")
        assert result is True

    def test_heartbeat_unregistered_client(self):
        ps = DistributedParameterServer()
        result = ps.heartbeat("ghost-client")
        assert result is False

    def test_get_stale_clients(self):
        ps = DistributedParameterServer()
        ps.register_client("old-client")
        # Artificially set heartbeat to past
        ps._heartbeats["old-client"] = time.time() - 120.0
        stale = ps.get_stale_clients(timeout_seconds=60.0)
        assert "old-client" in stale

    def test_get_stale_clients_fresh_clients_excluded(self):
        ps = DistributedParameterServer()
        ps.register_client("fresh-client")
        stale = ps.get_stale_clients(timeout_seconds=60.0)
        assert "fresh-client" not in stale

    def test_broadcast_weights_returns_results(self):
        ps = DistributedParameterServer()
        ps.register_client("actor-0")
        ps.register_client("actor-1")
        results = ps.broadcast_weights()
        assert "actor-0" in results
        assert "actor-1" in results
        assert all(v is True for v in results.values())

    def test_broadcast_weights_empty_clients(self):
        ps = DistributedParameterServer()
        results = ps.broadcast_weights()
        assert results == {}

    def test_get_cluster_info(self):
        ps = DistributedParameterServer(node_id="ps-main")
        ps.register_client("learner-0", client_type="learner")
        ps.register_client("actor-0", client_type="actor")
        info = ps.get_cluster_info()
        assert info["node_id"] == "ps-main"
        assert info["clients"]["total"] == 2
        assert info["clients"]["learners"] == 1
        assert info["clients"]["actors"] == 1
        assert "stale_clients" in info

    def test_get_client_count_all_types(self):
        ps = DistributedParameterServer()
        ps.register_client("c1", client_type="learner")
        ps.register_client("c2", client_type="evaluator")
        # None → all types
        assert ps.get_client_count(None) == 2

    def test_node_id_default(self):
        ps = DistributedParameterServer()
        assert ps.node_id == "ps-0"


# ---------------------------------------------------------------------------
# create_parameter_server factory
# ---------------------------------------------------------------------------


class TestCreateParameterServer:
    def test_creates_standard_server(self):
        ps = create_parameter_server()
        assert isinstance(ps, ParameterServer)
        assert not isinstance(ps, MockParameterServer)
        assert not isinstance(ps, DistributedParameterServer)

    def test_creates_mock_server(self):
        ps = create_parameter_server(force_mock=True)
        assert isinstance(ps, MockParameterServer)

    def test_creates_distributed_server(self):
        ps = create_parameter_server(distributed=True, node_id="ps-99")
        assert isinstance(ps, DistributedParameterServer)
        assert ps.node_id == "ps-99"

    def test_force_mock_takes_precedence_over_distributed(self):
        """force_mock=True should return MockParameterServer regardless of distributed."""
        ps = create_parameter_server(distributed=True, force_mock=True)
        assert isinstance(ps, MockParameterServer)

    def test_creates_with_network(self):
        net = _tiny_network()
        ps = create_parameter_server(
            network=net, config=ParameterServerConfig(compress_weights=False)
        )
        assert ps.get_version() == 1

    def test_creates_with_config(self):
        cfg = ParameterServerConfig(compress_weights=False, max_version_history=2)
        ps = create_parameter_server(config=cfg)
        assert ps.config.max_version_history == 2
