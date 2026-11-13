"""Unit tests for ``alpha_zero.serving.inference_server``.

Covers the FastAPI app surface (``/health``, ``/v1/model``, ``/v1/move``) and the
``_load_model`` startup path. The neural network and GCS download are mocked, so no
real model weights or network calls are required.

The serving module imports ``fastapi``/``uvicorn`` at module load time (the ``web``
extra). These tests ``importorskip`` those so they degrade to a skip when the extra
is not installed (e.g. the default CI ``[dev]`` job), rather than erroring on collection.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi", reason="serving server requires the 'web' extra (fastapi)")
pytest.importorskip("uvicorn", reason="serving server requires the 'web' extra (uvicorn)")

from fastapi.testclient import TestClient

import alpha_zero.serving.inference_server as srv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    """A TestClient that does NOT enter the lifespan context.

    Starlette only runs the ``lifespan`` startup (which calls ``_load_model``) when the
    client is used as a context manager. Instantiating it directly lets us exercise the
    endpoints with the global model state we set explicitly per-test.
    """
    return TestClient(srv.app)


@pytest.fixture
def reset_model_state():
    """Snapshot and restore the module-level model globals around a test."""
    saved = (srv._model, srv._model_version, srv._startup_time)
    yield
    srv._model, srv._model_version, srv._startup_time = saved


@pytest.fixture
def loaded_model(reset_model_state):
    """Install a mock network as the loaded model.

    The mock returns ``(policy_logits, value)`` matching the server's expected
    forward-pass contract for an 8x8 board / 376-action space.
    """
    policy_logits = torch.zeros(1, srv.ACTION_SPACE)
    value = torch.tensor([[0.25]])
    model = Mock(return_value=(policy_logits, value))
    srv._model = model
    srv._model_version = "v999"
    srv._startup_time = 1.5
    return model


def _valid_move_payload(temperature: float = 0.0) -> dict:
    """Build a structurally valid /v1/move request body."""
    mask = [0.0] * srv.ACTION_SPACE
    mask[0] = 1.0
    mask[1] = 1.0
    return {
        "spatial": np.zeros((13, srv.BOARD_SIZE, srv.BOARD_SIZE), dtype=np.float32).tolist(),
        "vector": np.zeros(64, dtype=np.float32).tolist(),
        "action_mask": mask,
        "temperature": temperature,
    }


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_reports_loading_when_no_model(client, reset_model_state):
    srv._model = None
    srv._model_version = ""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "loading"
    assert body["model_loaded"] is False
    assert body["device"]  # device string is populated


def test_health_reports_ok_when_model_loaded(client, loaded_model):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "v999"
    assert body["startup_time_s"] == pytest.approx(1.5, abs=0.01)


# ---------------------------------------------------------------------------
# /v1/model
# ---------------------------------------------------------------------------


def test_model_info_returns_config(client, loaded_model):
    resp = client.get("/v1/model")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == "v999"
    assert body["board_size"] == srv.BOARD_SIZE
    assert body["action_space_size"] == srv.ACTION_SPACE
    assert body["mcts_simulations"] == srv.MCTS_SIMS


# ---------------------------------------------------------------------------
# /v1/move
# ---------------------------------------------------------------------------


def test_move_returns_503_when_model_not_loaded(client, reset_model_state):
    srv._model = None
    resp = client.post("/v1/move", json=_valid_move_payload())
    assert resp.status_code == 503
    assert "not loaded" in resp.json()["detail"].lower()


def test_move_rejects_bad_spatial_shape(client, loaded_model):
    payload = _valid_move_payload()
    payload["spatial"] = np.zeros((13, 4, 4), dtype=np.float32).tolist()  # wrong board size
    resp = client.post("/v1/move", json=payload)
    assert resp.status_code == 422
    assert "spatial shape" in resp.json()["detail"]


def test_move_rejects_bad_vector_shape(client, loaded_model):
    payload = _valid_move_payload()
    payload["vector"] = [0.0] * 32  # should be 64
    resp = client.post("/v1/move", json=payload)
    assert resp.status_code == 422
    assert "vector shape" in resp.json()["detail"]


def test_move_rejects_bad_mask_shape(client, loaded_model):
    payload = _valid_move_payload()
    payload["action_mask"] = [1.0] * (srv.ACTION_SPACE - 1)  # one short
    resp = client.post("/v1/move", json=payload)
    assert resp.status_code == 422
    assert "action_mask shape" in resp.json()["detail"]


def test_move_greedy_selects_legal_action(client, loaded_model):
    # Make action index 1 dominate the logits so greedy selection is deterministic.
    logits = torch.full((1, srv.ACTION_SPACE), -10.0)
    logits[0, 1] = 10.0
    loaded_model.return_value = (logits, torch.tensor([[0.5]]))

    resp = client.post("/v1/move", json=_valid_move_payload(temperature=0.0))
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == 1  # the only high-logit legal action
    assert len(body["action_probabilities"]) == srv.ACTION_SPACE
    assert body["model_version"] == "v999"
    assert body["search_time_ms"] >= 0.0
    # Probability mass concentrates on the legal, high-logit action.
    assert body["action_probabilities"][1] > 0.5
    # Illegal actions get zero probability.
    assert body["action_probabilities"][5] == 0.0


def test_move_stochastic_selects_among_legal_actions(client, loaded_model):
    resp = client.post("/v1/move", json=_valid_move_payload(temperature=1.0))
    assert resp.status_code == 200
    body = resp.json()
    # Only indices 0 and 1 are legal in the payload mask.
    assert body["action"] in (0, 1)


def test_move_returns_500_on_internal_error(client, loaded_model):
    loaded_model.side_effect = RuntimeError("boom")
    resp = client.post("/v1/move", json=_valid_move_payload())
    assert resp.status_code == 500
    assert "boom" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# _load_model
# ---------------------------------------------------------------------------


def _fake_arch() -> Mock:
    """A NetworkArchitectureConfig-shaped mock with valid integer fields."""
    return Mock(
        spatial_channels=13,
        board_size=8,
        vector_dim=64,
        action_space_size=376,
        embed_dim=8,
        num_layers=1,
        num_heads=1,
        policy_hidden_dim=8,
        value_head_dim=8,
    )


def _network_patches(fake_net: Mock):
    return (
        patch(
            "alpha_zero.cloud.gcp.config.NetworkArchitectureConfig.from_env",
            return_value=_fake_arch(),
        ),
        patch(
            "alpha_zero.core.transformer_network.Civ6TransformerNetwork",
            return_value=fake_net,
        ),
        patch("alpha_zero.core.transformer_network.TransformerConfig"),
    )


def _install_failing_gcs(monkeypatch) -> None:
    """A ``google.cloud.storage`` whose client cannot be built (no credentials)."""
    fake_storage = types.ModuleType("google.cloud.storage")
    fake_storage.Client = Mock(side_effect=OSError("no credentials"))
    fake_cloud = types.ModuleType("google.cloud")
    fake_cloud.storage = fake_storage
    fake_google = types.ModuleType("google")
    fake_google.cloud = fake_cloud
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.cloud", fake_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", fake_storage)


def test_load_model_refuses_to_start_without_a_bucket(reset_model_state, monkeypatch):
    """No bucket and no opt-in: fail closed, never serve random weights.

    The previous default was the maintainer's bucket name. A bucket-name default
    is a takeover vector (whoever registers the name serves the model), and the
    old fallback would then have served a randomly initialised network behind a
    green health check.
    """
    srv._model = None
    monkeypatch.setattr(srv, "GCS_BUCKET", None)
    monkeypatch.setattr(srv, "ALLOW_RANDOM_WEIGHTS", False)
    fake_net = Mock()
    fake_net.to.return_value = fake_net

    p1, p2, p3 = _network_patches(fake_net)
    with p1, p2, p3, pytest.raises(RuntimeError, match="CIV6AZ_GCS_BUCKET is not set"):
        srv._load_model()

    assert srv._model is None


def test_load_model_refuses_random_weights_when_download_fails(reset_model_state, monkeypatch):
    """A configured bucket whose download fails is an error, not a silent downgrade."""
    srv._model = None
    monkeypatch.setattr(srv, "GCS_BUCKET", "some-bucket")
    monkeypatch.setattr(srv, "ALLOW_RANDOM_WEIGHTS", False)
    _install_failing_gcs(monkeypatch)
    fake_net = Mock()
    fake_net.to.return_value = fake_net

    p1, p2, p3 = _network_patches(fake_net)
    with p1, p2, p3, pytest.raises(RuntimeError, match="Refusing to serve random weights"):
        srv._load_model()

    assert srv._model is None
    fake_net.load_state_dict.assert_not_called()


def test_load_model_serves_random_weights_only_on_explicit_opt_in(reset_model_state, monkeypatch):
    """``CIV6AZ_ALLOW_RANDOM_WEIGHTS=1`` is the one path to random weights."""
    srv._model = None
    monkeypatch.setattr(srv, "GCS_BUCKET", None)
    monkeypatch.setattr(srv, "ALLOW_RANDOM_WEIGHTS", True)
    fake_net = Mock()
    fake_net.to.return_value = fake_net

    p1, p2, p3 = _network_patches(fake_net)
    with p1, p2, p3:
        srv._load_model()

    assert srv._model is fake_net
    assert srv._model_version == srv.MODEL_VERSION
    assert srv._startup_time >= 0.0
    fake_net.to.assert_called_once()
    fake_net.load_state_dict.assert_not_called()


def test_load_model_loads_state_dict_when_gcs_succeeds(reset_model_state, monkeypatch):
    """Cover the GCS-download + state_dict load branch with fully mocked I/O."""
    srv._model = None
    monkeypatch.setattr(srv, "GCS_BUCKET", "some-bucket")
    fake_net = Mock()
    fake_net.to.return_value = fake_net

    # Build a fake google.cloud.storage module tree so the in-function import resolves.
    blob = Mock()
    bucket = Mock()
    bucket.blob.return_value = blob
    gcs_client = Mock()
    gcs_client.bucket.return_value = bucket

    fake_storage = types.ModuleType("google.cloud.storage")
    fake_storage.Client = Mock(return_value=gcs_client)
    fake_cloud = types.ModuleType("google.cloud")
    fake_cloud.storage = fake_storage
    fake_google = types.ModuleType("google")
    fake_google.cloud = fake_cloud

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.cloud", fake_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", fake_storage)

    with (
        patch(
            "alpha_zero.cloud.gcp.config.NetworkArchitectureConfig.from_env",
            return_value=_fake_arch(),
        ),
        patch(
            "alpha_zero.core.transformer_network.Civ6TransformerNetwork",
            return_value=fake_net,
        ),
        patch("alpha_zero.core.transformer_network.TransformerConfig"),
        patch.object(srv.torch, "load", return_value={"w": 1}) as torch_load,
    ):
        srv._load_model()

    blob.download_to_filename.assert_called_once()
    torch_load.assert_called_once()
    fake_net.load_state_dict.assert_called_once_with({"w": 1})
    assert srv._model is fake_net
