"""Coverage tests for alpha_zero.core.checkpoint_utils module.

The ``TestNoAutomaticUnsafeFallback`` class pins WS4.1 of the tech-debt
hardening spec: ``load_checkpoint`` used to catch *any* exception from the
``weights_only=True`` load and silently retry with ``weights_only=False``, so a
malicious or merely legacy pickle was always executed. Those tests are written
so the **old** code would have passed the negative-control case silently -- the
non-tensor payload loaded fine -- and only the new code raises.
"""

import io
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from alpha_zero.core.checkpoint_utils import (
    UNSAFE_CHECKPOINT_ENV_VAR,
    VALID_REDUCTIONS,
    UnsafeCheckpointError,
    _convert_to_primitives,
    get_checkpoint_info,
    load_checkpoint,
    load_state_dict_into_network,
    safe_torch_load,
    save_checkpoint,
    serialize_config,
    unsafe_checkpoint_loading_allowed,
    validate_reduction,
)


class SampleColor(Enum):
    RED = "red"
    BLUE = "blue"


@dataclass
class SampleConfig:
    lr: float = 0.001
    hidden_dim: int = 64
    name: str = "test"


@dataclass
class NestedConfig:
    inner: SampleConfig = None
    values: list = None

    def __post_init__(self):
        if self.inner is None:
            self.inner = SampleConfig()
        if self.values is None:
            self.values = [1, 2, 3]


class TestSerializeConfig:
    def test_basic_dataclass(self):
        config = SampleConfig(lr=0.01, hidden_dim=128)
        result = serialize_config(config)
        assert result["lr"] == 0.01
        assert result["hidden_dim"] == 128
        assert result["name"] == "test"

    def test_not_dataclass_raises(self):
        with pytest.raises(TypeError, match="config must be a dataclass"):
            serialize_config({"lr": 0.01})

    def test_class_not_instance_raises(self):
        with pytest.raises(TypeError, match="config must be a dataclass instance"):
            serialize_config(SampleConfig)

    def test_nested_dataclass(self):
        config = NestedConfig()
        result = serialize_config(config)
        assert result["inner"]["lr"] == 0.001
        assert result["values"] == [1, 2, 3]


class TestConvertToPrimitives:
    def test_dict(self):
        result = _convert_to_primitives({"a": 1, "b": "hello"})
        assert result == {"a": 1, "b": "hello"}

    def test_list(self):
        result = _convert_to_primitives([1, 2, 3])
        assert result == [1, 2, 3]

    def test_tuple(self):
        result = _convert_to_primitives((1, 2))
        assert result == [1, 2]

    def test_enum(self):
        result = _convert_to_primitives(SampleColor.RED)
        assert result == "red"

    def test_numpy_scalar(self):
        result = _convert_to_primitives(np.float32(3.14))
        assert isinstance(result, float)

    def test_numpy_scalar_via_item(self):
        # numpy scalars have .item(), which takes priority over .tolist()
        result = _convert_to_primitives(np.int64(42))
        assert result == 42

    def test_nested(self):
        data = {"color": SampleColor.BLUE, "values": [np.int32(1), np.float64(2.0)]}
        result = _convert_to_primitives(data)
        assert result["color"] == "blue"
        assert result["values"] == [1, 2.0]

    def test_plain_value(self):
        assert _convert_to_primitives(42) == 42
        assert _convert_to_primitives("hello") == "hello"


class TestSaveCheckpoint:
    def test_basic_save(self, tmp_path):
        net = nn.Linear(10, 5)
        config = SampleConfig()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(net, config, path)
        assert path.exists()

    def test_save_with_network_type(self, tmp_path):
        net = nn.Linear(10, 5)
        config = SampleConfig()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(net, config, path, network_type="test_v1")
        ckpt = torch.load(path, weights_only=False)
        assert ckpt["network_type"] == "test_v1"

    def test_save_with_extra_data(self, tmp_path):
        net = nn.Linear(10, 5)
        config = SampleConfig()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(net, config, path, extra_data={"epoch": 10, "loss": 0.5})
        ckpt = torch.load(path, weights_only=False)
        assert ckpt["epoch"] == 10
        assert ckpt["loss"] == 0.5

    def test_save_creates_parent_dirs(self, tmp_path):
        net = nn.Linear(10, 5)
        config = SampleConfig()
        path = tmp_path / "deep" / "nested" / "ckpt.pt"
        save_checkpoint(net, config, path)
        assert path.exists()

    def test_save_string_path(self, tmp_path):
        net = nn.Linear(10, 5)
        config = SampleConfig()
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(net, config, path)
        assert Path(path).exists()


class TestLoadCheckpoint:
    def test_basic_load(self, tmp_path):
        net = nn.Linear(10, 5)
        config = SampleConfig()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(net, config, path)
        ckpt = load_checkpoint(path)
        assert "config" in ckpt
        assert "state_dict" in ckpt

    def test_load_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.pt")

    def test_load_string_path(self, tmp_path):
        net = nn.Linear(10, 5)
        config = SampleConfig()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(net, config, path)
        ckpt = load_checkpoint(str(path))
        assert "state_dict" in ckpt

    def test_load_with_device(self, tmp_path):
        net = nn.Linear(10, 5)
        config = SampleConfig()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(net, config, path)
        ckpt = load_checkpoint(path, device="cpu")
        assert "state_dict" in ckpt

    def test_load_weights_only_false(self, tmp_path):
        net = nn.Linear(10, 5)
        config = SampleConfig()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(net, config, path)
        ckpt = load_checkpoint(path, weights_only=False)
        assert "state_dict" in ckpt

    def test_load_weights_only_false_warns_deprecated(self, tmp_path, caplog):
        """``weights_only=False`` still works but is logged as a deprecated alias."""
        net = nn.Linear(10, 5)
        path = tmp_path / "ckpt.pt"
        save_checkpoint(net, SampleConfig(), path)
        with caplog.at_level(logging.WARNING, logger="alpha_zero.core.checkpoint_utils"):
            ckpt = load_checkpoint(path, weights_only=False)
        assert "state_dict" in ckpt
        assert any("deprecated" in rec.getMessage() for rec in caplog.records)


def _write_non_tensor_payload(path: Path) -> None:
    """Write a checkpoint that ``weights_only=True`` rejects.

    A dataclass *instance* is not in torch's safe-globals allowlist, so the
    tensor-only unpickler refuses it. It stands in for the class of payload the
    old auto-downgrade silently executed.
    """
    torch.save({"config": SampleConfig(), "state_dict": nn.Linear(2, 2).state_dict()}, path)


class TestNoAutomaticUnsafeFallback:
    """WS4.1: safe-load failure raises unless unsafe loading is explicitly authorised."""

    @pytest.fixture(autouse=True)
    def _clear_env(self, monkeypatch):
        monkeypatch.delenv(UNSAFE_CHECKPOINT_ENV_VAR, raising=False)

    def test_safe_load_works_for_tensor_payload(self, tmp_path):
        net = nn.Linear(10, 5)
        path = tmp_path / "ckpt.pt"
        save_checkpoint(net, SampleConfig(), path)
        ckpt = load_checkpoint(path)
        assert set(ckpt) >= {"config", "state_dict"}

    def test_non_tensor_payload_raises_without_flag(self, tmp_path):
        """Negative control for the removed fallback.

        Under the pre-WS4.1 code this test would have passed *silently*: the
        ``except Exception`` branch retried with ``weights_only=False`` and the
        dataclass instance came back as a normal value. The assertion therefore
        only holds because the downgrade is gone.
        """
        path = tmp_path / "legacy.pt"
        _write_non_tensor_payload(path)
        with pytest.raises(UnsafeCheckpointError) as excinfo:
            load_checkpoint(path)
        message = str(excinfo.value)
        assert str(path) in message
        assert "allow_unsafe=True" in message
        assert UNSAFE_CHECKPOINT_ENV_VAR in message
        # The original torch rejection is chained, not swallowed.
        assert excinfo.value.__cause__ is not None

    def test_non_tensor_payload_loads_with_flag_and_warns(self, tmp_path, caplog):
        path = tmp_path / "legacy.pt"
        _write_non_tensor_payload(path)
        with caplog.at_level(logging.WARNING, logger="alpha_zero.core.checkpoint_utils"):
            ckpt = load_checkpoint(path, allow_unsafe=True)
        assert isinstance(ckpt["config"], SampleConfig)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "unsafe load must be logged at WARNING"
        assert "weights_only=False" in warnings[0].getMessage()
        assert "allow_unsafe=True" in warnings[0].getMessage()

    @pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
    def test_env_var_authorises_unsafe_load(self, tmp_path, monkeypatch, caplog, value):
        path = tmp_path / "legacy.pt"
        _write_non_tensor_payload(path)
        monkeypatch.setenv(UNSAFE_CHECKPOINT_ENV_VAR, value)
        with caplog.at_level(logging.WARNING, logger="alpha_zero.core.checkpoint_utils"):
            ckpt = load_checkpoint(path)
        assert isinstance(ckpt["config"], SampleConfig)
        assert any(UNSAFE_CHECKPOINT_ENV_VAR in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("value", ["0", "false", "", "no"])
    def test_env_var_falsy_values_do_not_authorise(self, tmp_path, monkeypatch, value):
        path = tmp_path / "legacy.pt"
        _write_non_tensor_payload(path)
        monkeypatch.setenv(UNSAFE_CHECKPOINT_ENV_VAR, value)
        with pytest.raises(UnsafeCheckpointError):
            load_checkpoint(path)

    def test_env_var_is_read_at_call_time(self, tmp_path, monkeypatch):
        """Setting the variable after import must take effect (no import-time snapshot)."""
        path = tmp_path / "legacy.pt"
        _write_non_tensor_payload(path)
        assert unsafe_checkpoint_loading_allowed() is False
        with pytest.raises(UnsafeCheckpointError):
            load_checkpoint(path)
        monkeypatch.setenv(UNSAFE_CHECKPOINT_ENV_VAR, "1")
        assert unsafe_checkpoint_loading_allowed() is True
        assert "config" in load_checkpoint(path)

    def test_safe_torch_load_resets_buffer_before_unsafe_retry(self):
        """A file object consumed by the failed safe attempt is rewound for the retry."""
        buffer = io.BytesIO()
        torch.save({"config": SampleConfig()}, buffer)
        buffer.seek(0)
        with pytest.raises(UnsafeCheckpointError):
            safe_torch_load(buffer)
        buffer.seek(0)
        loaded = safe_torch_load(buffer, allow_unsafe=True)
        assert isinstance(loaded["config"], SampleConfig)

    def test_get_checkpoint_info_respects_flag(self, tmp_path):
        path = tmp_path / "legacy.pt"
        _write_non_tensor_payload(path)
        with pytest.raises(UnsafeCheckpointError):
            get_checkpoint_info(path)
        info = get_checkpoint_info(path, allow_unsafe=True)
        assert "state_dict" in info["keys"]


class TestLoadStateDictIntoNetwork:
    def test_basic_load(self):
        net = nn.Linear(10, 5)
        # Save and create checkpoint dict
        state_dict = net.state_dict()
        checkpoint = {"state_dict": state_dict}

        net2 = nn.Linear(10, 5)
        result = load_state_dict_into_network(net2, checkpoint)
        assert result is net2

        # Verify weights match
        for k in state_dict:
            assert torch.equal(net2.state_dict()[k], state_dict[k])

    def test_load_without_state_dict_key(self):
        """When checkpoint IS the state dict."""
        net = nn.Linear(10, 5)
        state_dict = net.state_dict()

        net2 = nn.Linear(10, 5)
        result = load_state_dict_into_network(net2, state_dict)
        assert result is net2

    def test_non_strict_load(self):
        net = nn.Linear(10, 5)
        state_dict = net.state_dict()
        # Add extra key
        state_dict["extra_key"] = torch.zeros(1)
        checkpoint = {"state_dict": state_dict}

        net2 = nn.Linear(10, 5)
        # strict=False should not raise
        load_state_dict_into_network(net2, checkpoint, strict=False)


class TestValidateReduction:
    def test_valid_reductions(self):
        for r in ["mean", "sum", "none"]:
            validate_reduction(r)  # should not raise

    def test_invalid_reduction(self):
        with pytest.raises(ValueError, match="Unknown reduction"):
            validate_reduction("invalid")

    def test_valid_reductions_constant(self):
        assert VALID_REDUCTIONS == frozenset({"mean", "sum", "none"})


class TestGetCheckpointInfo:
    def test_basic_info(self, tmp_path):
        net = nn.Linear(10, 5)
        config = SampleConfig()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(net, config, path, network_type="test_v1")

        info = get_checkpoint_info(path)
        assert info["path"] == str(path)
        assert info["file_size_mb"] > 0
        assert "config" in info["keys"]
        assert "state_dict" in info["keys"]
        assert info["network_type"] == "test_v1"
        assert "config_keys" in info
        assert "num_parameters" in info

    def test_info_without_network_type(self, tmp_path):
        net = nn.Linear(10, 5)
        config = SampleConfig()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(net, config, path)

        info = get_checkpoint_info(path)
        assert "network_type" not in info

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_checkpoint_info(tmp_path / "nonexistent.pt")

    def test_num_parameters(self, tmp_path):
        net = nn.Linear(10, 5, bias=True)
        config = SampleConfig()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(net, config, path)

        info = get_checkpoint_info(path)
        # Linear(10, 5) has 10*5 + 5 = 55 parameters
        assert info["num_parameters"] == 55


class TestSaveLoadRoundtrip:
    def test_full_roundtrip(self, tmp_path):
        net = nn.Linear(10, 5)
        config = SampleConfig(lr=0.01, hidden_dim=256, name="roundtrip")
        path = tmp_path / "roundtrip.pt"

        save_checkpoint(net, config, path, network_type="test", extra_data={"step": 100})

        ckpt = load_checkpoint(path)
        assert ckpt["config"]["lr"] == 0.01
        assert ckpt["config"]["hidden_dim"] == 256
        assert ckpt["network_type"] == "test"
        assert ckpt["step"] == 100

        net2 = nn.Linear(10, 5)
        load_state_dict_into_network(net2, ckpt)

        # Verify weights match
        for k in net.state_dict():
            assert torch.equal(net.state_dict()[k], net2.state_dict()[k])
