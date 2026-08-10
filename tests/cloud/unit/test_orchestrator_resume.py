"""Resume-from-checkpoint token resolution for the Vertex orchestrator."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock

import pytest

from alpha_zero.cloud.gcp import vertex_orchestrator as vo


def _make_args(**overrides) -> argparse.Namespace:
    # Resume-token resolver only needs model_version_start, start_iteration,
    # gcs_bucket, resume_from.  build_training_extra_args needs the full
    # network/LR set — include sensible defaults so the two arg-builder tests
    # exercise real code, not stray AttributeError paths.
    base = {
        "model_version_start": 0,
        "start_iteration": 0,
        "gcs_bucket": "test-bucket",
        "resume_from": None,
        # Network architecture (matches CLI defaults)
        "board_size": 8,
        "action_space_size": 376,
        "num_res_blocks": 6,
        "num_filters": 64,
        "num_heads": 8,
        "policy_hidden_dim": 128,
        "value_head_dim": 64,
        # LR schedule (defaults are no-ops in builder except learning_rate)
        "lr_schedule_type": "constant",
        "lr_initial": 0.001,
        "lr_final": 0.0001,
        "lr_warmup_steps": 0,
        "lr_decay_steps": 0,
        "weight_decay": 0.01,
        "value_lr_multiplier": 1.0,
        "policy_lr_multiplier": 1.0,
        "learning_rate": 0.001,
        "algorithm": "alphazero",
        "encoder_type": "patch_embed",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_resolve_returns_none_when_no_incumbent_checkpoint():
    """Iteration 0 with no GCS checkpoint -> token is None (cold start)."""
    fake_ckpt_storage = MagicMock()
    fake_ckpt_storage.load_checkpoint.return_value = None  # no latest
    args = _make_args()

    result = vo._resolve_resume_token(args, iteration=0, checkpoint_storage=fake_ckpt_storage)

    assert result is None


def test_resolve_returns_latest_when_incumbent_exists():
    """Iteration N>0 with a saved checkpoint -> token is 'latest'."""
    fake_ckpt_storage = MagicMock()
    fake_ckpt_storage.load_checkpoint.return_value = {"step": 500}
    args = _make_args()

    result = vo._resolve_resume_token(args, iteration=3, checkpoint_storage=fake_ckpt_storage)

    assert result == "latest"


def test_explicit_args_resume_from_overrides_auto_resolution():
    """When --resume_from is set on the CLI, it wins."""
    fake_ckpt_storage = MagicMock()
    args = _make_args(resume_from="42")
    result = vo._resolve_resume_token(args, iteration=3, checkpoint_storage=fake_ckpt_storage)
    assert result == "42"
    fake_ckpt_storage.load_checkpoint.assert_not_called()


def test_resolve_returns_none_on_gcs_storage_error():
    """Expected: GCSStorageError (missing checkpoint) -> cold-start (None)."""
    from alpha_zero.cloud.gcp.exceptions import GCSStorageError

    fake_ckpt_storage = MagicMock()
    fake_ckpt_storage.load_checkpoint.side_effect = GCSStorageError("not found")
    args = _make_args()

    result = vo._resolve_resume_token(args, iteration=3, checkpoint_storage=fake_ckpt_storage)
    assert result is None


def test_resolve_reraises_unexpected_exceptions():
    """Unexpected errors (not GCSStorageError) must surface, not silently cold-start.

    Reproduces the v1.2.0 zero-promotion failure mode: a misconfigured
    GCPConfig or a torch.load deserialisation crash used to be swallowed
    by a broad ``except Exception`` and treated as "no checkpoint", causing
    every iteration to train from random init.
    """
    fake_ckpt_storage = MagicMock()
    fake_ckpt_storage.load_checkpoint.side_effect = RuntimeError("torch deser failure")
    args = _make_args()

    with pytest.raises(RuntimeError, match="torch deser failure"):
        vo._resolve_resume_token(args, iteration=3, checkpoint_storage=fake_ckpt_storage)


def test_build_training_extra_args_emits_resume_when_provided():
    args = _make_args(resume_from="latest")
    extra = vo.build_training_extra_args(args)
    assert "--resume_from=latest" in extra


def test_build_training_extra_args_omits_resume_when_none():
    args = _make_args(resume_from=None)
    extra = vo.build_training_extra_args(args)
    assert not any(a.startswith("--resume_from") for a in extra)


def test_build_training_extra_args_does_not_emit_model_version():
    """training_worker.py no longer accepts --model_version; resumption is via
    --resume_from. The orchestrator must not emit --model_version even when
    a model_version kwarg is passed (kept on the signature for caller compat).
    """
    args = _make_args()
    extra_no_kwarg = vo.build_training_extra_args(args)
    extra_with_kwarg = vo.build_training_extra_args(args, model_version=42)
    for extra in (extra_no_kwarg, extra_with_kwarg):
        assert not any(a.startswith("--model_version") for a in extra), (
            f"--model_version must not be emitted; got {extra!r}"
        )


def test_resolve_resume_honors_cli_bucket_override(monkeypatch):
    """When --gcs_bucket is set on args, the helper builds GCPConfig with
    that bucket rather than silently falling through to CIV6AZ_GCS_BUCKET.
    """
    monkeypatch.delenv("CIV6AZ_GCS_BUCKET", raising=False)
    captured: dict[str, object] = {}

    class _FakeGCPConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.bucket = kwargs.get("bucket", "from-env")
            self.project_id = kwargs.get("project_id", "p")
            self.region = kwargs.get("region", "us-central1")

    class _FakeCkptStorage:
        def __init__(self, config):
            self.config = config

        def load_checkpoint(self, step=None):
            return None

    # Patch the name as imported in vertex_orchestrator (module-level)
    monkeypatch.setattr(vo, "GCPConfig", _FakeGCPConfig)
    monkeypatch.setattr("alpha_zero.cloud.gcp.storage.GCSCheckpointStorage", _FakeCkptStorage)

    args = _make_args(gcs_bucket="cli-bucket-override")
    # Add the optional CLI fields the helper reads via getattr
    args.project = "cli-project"
    args.region = "us-east1"

    result = vo._resolve_resume_token(args, iteration=0)
    assert result is None  # no checkpoint -> cold start
    assert captured.get("bucket") == "cli-bucket-override"
    assert captured.get("project_id") == "cli-project"
    assert captured.get("region") == "us-east1"
