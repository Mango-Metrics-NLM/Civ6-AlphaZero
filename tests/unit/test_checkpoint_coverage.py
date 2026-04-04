"""Comprehensive tests for alpha_zero.utils.checkpoint module."""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from alpha_zero.utils.checkpoint import (
    CheckpointLoadResult,
    compute_weight_checksum,
    load_model_state,
    safe_load_checkpoint,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# CheckpointLoadResult
# ---------------------------------------------------------------------------


class TestCheckpointLoadResult:
    def test_creation(self):
        result = CheckpointLoadResult(
            data={"key": "value"},
            loaded_safely=True,
            warnings=[],
        )
        assert result.data == {"key": "value"}
        assert result.loaded_safely is True
        assert result.warnings == []

    def test_with_warnings(self):
        result = CheckpointLoadResult(
            data={},
            loaded_safely=False,
            warnings=["loaded unsafely"],
        )
        assert len(result.warnings) == 1


# ---------------------------------------------------------------------------
# safe_load_checkpoint
# ---------------------------------------------------------------------------


class TestSafeLoadCheckpoint:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            safe_load_checkpoint(tmp_path / "missing.pt")

    def test_safe_load_success(self, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        data = {"model_state_dict": {"weight": torch.tensor([1.0, 2.0])}}
        torch.save(data, ckpt_path)

        result = safe_load_checkpoint(ckpt_path, map_location="cpu")
        assert result.loaded_safely is True
        assert "model_state_dict" in result.data
        assert result.warnings == []

    def test_safe_load_with_string_path(self, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        torch.save({"a": torch.tensor([1.0])}, ckpt_path)
        result = safe_load_checkpoint(str(ckpt_path), map_location="cpu")
        assert result.loaded_safely is True

    @patch("torch.load")
    def test_safe_load_fallback_to_unsafe(self, mock_load, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        ckpt_path.write_bytes(b"fake checkpoint data")

        # First call (weights_only=True) fails, second (weights_only=False) succeeds
        mock_load.side_effect = [
            RuntimeError("weights_only failed"),
            {"model_state_dict": {"w": "data"}},
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = safe_load_checkpoint(ckpt_path, map_location="cpu")

        assert result.loaded_safely is False
        assert len(result.warnings) == 1
        assert "weights_only=False" in result.warnings[0]
        assert len(w) >= 1

    @patch("torch.load")
    def test_strict_security_raises(self, mock_load, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        ckpt_path.write_bytes(b"fake data")
        mock_load.side_effect = RuntimeError("weights_only failed")

        with pytest.raises(RuntimeError, match="Cannot load checkpoint safely"):
            safe_load_checkpoint(ckpt_path, strict_security=True)

    @patch("torch.load")
    def test_fallback_disabled_raises(self, mock_load, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        ckpt_path.write_bytes(b"fake data")
        mock_load.side_effect = RuntimeError("weights_only failed")

        with pytest.raises(RuntimeError, match="fallback is disabled"):
            safe_load_checkpoint(
                ckpt_path,
                strict_security=False,
                allow_unsafe_fallback=False,
            )

    def test_map_location_device(self, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        torch.save({"w": torch.tensor([1.0])}, ckpt_path)
        result = safe_load_checkpoint(ckpt_path, map_location=torch.device("cpu"))
        assert result.loaded_safely is True


# ---------------------------------------------------------------------------
# load_model_state
# ---------------------------------------------------------------------------


class TestLoadModelState:
    def test_load_with_model_state_dict_key(self, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        state = {"layer.weight": torch.tensor([1.0, 2.0])}
        torch.save({"model_state_dict": state}, ckpt_path)

        loaded = load_model_state(ckpt_path, map_location="cpu")
        assert "layer.weight" in loaded

    def test_load_with_network_state_dict_key(self, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        state = {"layer.weight": torch.tensor([3.0])}
        torch.save({"network_state_dict": state}, ckpt_path)

        loaded = load_model_state(ckpt_path, map_location="cpu")
        assert "layer.weight" in loaded

    def test_load_with_state_dict_key(self, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        state = {"layer.bias": torch.tensor([0.5])}
        torch.save({"state_dict": state}, ckpt_path)

        loaded = load_model_state(ckpt_path, map_location="cpu")
        assert "layer.bias" in loaded

    def test_load_direct_state_dict(self, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        state = {"layer.weight": torch.tensor([1.0]), "layer.bias": torch.tensor([0.0])}
        torch.save(state, ckpt_path)

        loaded = load_model_state(ckpt_path, map_location="cpu")
        assert "layer.weight" in loaded

    def test_load_no_recognized_key_raises(self, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        # Save data where values are not all tensors and keys don't match
        torch.save({"config": {"lr": 0.001}, "epoch": 10}, ckpt_path)

        with pytest.raises(KeyError, match="Could not find model state dict"):
            load_model_state(ckpt_path, map_location="cpu")


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def test_save_basic(self, tmp_path):
        ckpt_path = tmp_path / "checkpoint.pt"
        state = {"layer.weight": torch.tensor([1.0])}
        save_checkpoint(ckpt_path, model_state_dict=state)

        assert ckpt_path.exists()
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "model_state_dict" in loaded

    def test_save_with_optimizer(self, tmp_path):
        ckpt_path = tmp_path / "checkpoint.pt"
        model_state = {"w": torch.tensor([1.0])}
        opt_state = {"state": {}, "param_groups": []}
        save_checkpoint(ckpt_path, model_state_dict=model_state, optimizer_state_dict=opt_state)

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "optimizer_state_dict" in loaded

    def test_save_with_config(self, tmp_path):
        ckpt_path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            ckpt_path,
            model_state_dict={"w": torch.tensor([1.0])},
            config={"lr": 0.001},
        )

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert loaded["config"] == {"lr": 0.001}

    def test_save_with_extra_data(self, tmp_path):
        ckpt_path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            ckpt_path,
            model_state_dict={"w": torch.tensor([1.0])},
            epoch=10,
            best_score=0.95,
        )

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert loaded["epoch"] == 10
        assert loaded["best_score"] == 0.95

    def test_save_creates_parent_dirs(self, tmp_path):
        ckpt_path = tmp_path / "deep" / "nested" / "dir" / "checkpoint.pt"
        save_checkpoint(ckpt_path, model_state_dict={"w": torch.tensor([1.0])})
        assert ckpt_path.exists()

    def test_save_string_path(self, tmp_path):
        ckpt_path = str(tmp_path / "model.pt")
        save_checkpoint(ckpt_path, model_state_dict={"w": torch.tensor([1.0])})
        assert Path(ckpt_path).exists()

    def test_save_without_optional_fields(self, tmp_path):
        ckpt_path = tmp_path / "bare.pt"
        save_checkpoint(ckpt_path, model_state_dict={"w": torch.tensor([0.0])})

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" not in loaded
        assert "config" not in loaded


# ---------------------------------------------------------------------------
# compute_weight_checksum
# ---------------------------------------------------------------------------


class TestComputeWeightChecksum:
    def test_simple_model(self):
        model = torch.nn.Linear(4, 2, bias=False)
        checksum = compute_weight_checksum(model)
        assert isinstance(checksum, float)
        assert checksum >= 0.0

    def test_model_with_bias(self):
        model = torch.nn.Linear(4, 2, bias=True)
        checksum = compute_weight_checksum(model)
        assert checksum > 0.0

    def test_zero_model(self):
        model = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        checksum = compute_weight_checksum(model)
        assert checksum == 0.0

    def test_deterministic(self):
        model = torch.nn.Linear(4, 2)
        c1 = compute_weight_checksum(model)
        c2 = compute_weight_checksum(model)
        assert c1 == c2

    def test_different_weights_different_checksum(self):
        m1 = torch.nn.Linear(4, 2, bias=False)
        m2 = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            m1.weight.fill_(1.0)
            m2.weight.fill_(2.0)
        assert compute_weight_checksum(m1) != compute_weight_checksum(m2)

    def test_sequential_model(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 2),
        )
        checksum = compute_weight_checksum(model)
        assert checksum > 0.0

    def test_empty_model(self):
        # A module with no parameters
        model = torch.nn.ReLU()
        checksum = compute_weight_checksum(model)
        assert checksum == 0.0


# ---------------------------------------------------------------------------
# Roundtrip: save then load
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    def test_full_roundtrip(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        original_state = model.state_dict()
        ckpt_path = tmp_path / "roundtrip.pt"

        save_checkpoint(ckpt_path, model_state_dict=original_state, epoch=5)
        result = safe_load_checkpoint(ckpt_path, map_location="cpu")

        assert result.loaded_safely is True
        assert result.data["epoch"] == 5
        loaded_state = result.data["model_state_dict"]

        for key in original_state:
            assert torch.equal(original_state[key], loaded_state[key])

    def test_load_model_state_roundtrip(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        ckpt_path = tmp_path / "roundtrip2.pt"
        save_checkpoint(ckpt_path, model_state_dict=model.state_dict())

        loaded_state = load_model_state(ckpt_path, map_location="cpu")

        new_model = torch.nn.Linear(4, 2)
        new_model.load_state_dict(loaded_state)

        for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=False):
            assert torch.equal(p1, p2)
