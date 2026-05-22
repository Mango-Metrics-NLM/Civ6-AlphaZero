"""End-to-end continuity test: save then load via GCSCheckpointStorage.

Locks the contract underlying vertex_orchestrator._resolve_resume_token:
- save_checkpoint writes both numbered and latest pointer
- load_checkpoint(step=None) reads the latest pointer
- weights round-trip bit-exact
"""
from __future__ import annotations

import torch

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.storage import GCSCheckpointStorage
from alpha_zero.cloud.gcp.training_worker import create_model
from tests.mocks.cloud_mocks import MockStorageBackend


def _make_config() -> GCPConfig:
    """Build the minimal GCPConfig needed by GCSCheckpointStorage."""
    return GCPConfig(project_id="p", bucket="b", region="us-central1")


def test_checkpoint_round_trip_preserves_weights():
    """save_checkpoint then load_checkpoint(step=None) must restore weights bit-exact."""
    ckpt_store = GCSCheckpointStorage(config=_make_config(), backend=MockStorageBackend())

    model_a = create_model(
        board_size=8,
        action_space_size=376,
        num_res_blocks=2,
        num_filters=64,
    )
    optimizer = torch.optim.Adam(model_a.parameters(), lr=1e-3)

    ckpt_store.save_checkpoint(model=model_a, optimizer=optimizer, step=42)

    loaded = ckpt_store.load_checkpoint(step=None)  # 'latest' pointer
    assert loaded is not None
    assert loaded["step"] == 42

    model_b = create_model(
        board_size=8,
        action_space_size=376,
        num_res_blocks=2,
        num_filters=64,
    )
    model_b.load_state_dict(loaded["model_state_dict"])

    for key, tensor_a in model_a.state_dict().items():
        tensor_b = model_b.state_dict()[key]
        assert torch.equal(tensor_a, tensor_b), f"mismatch at {key}"


def test_load_checkpoint_returns_none_when_absent():
    """Cold start: backend has no blobs, load returns None."""
    ckpt_store = GCSCheckpointStorage(config=_make_config(), backend=MockStorageBackend())
    assert ckpt_store.load_checkpoint(step=None) is None
