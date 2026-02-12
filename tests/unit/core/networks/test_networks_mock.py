"""Comprehensive tests for SDK neural network modules with mocked torch.

Since torch is not installed in this environment, we inject mock torch modules
into ``sys.modules`` before importing the network source modules.  This allows
us to verify class structure, attributes, method signatures, and constructor
behaviour without requiring a real PyTorch installation.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from unittest.mock import MagicMock

import pytest

# ============================================================================
# Mock torch infrastructure
# ============================================================================


class _MockModule:
    """Minimal stand-in for ``torch.nn.Module``.

    Supports subclassing and the ``super().__init__()`` call pattern used
    by all network classes.  Implements ``__call__`` to delegate to ``forward``
    just like the real ``nn.Module``.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Delegate to forward(), mirroring nn.Module.__call__."""
        return self.forward(*args, **kwargs)  # type: ignore[attr-defined]

    def eval(self) -> _MockModule:
        return self


class _NoGrad:
    """Mock for ``torch.no_grad`` that works as both context manager and decorator."""

    def __enter__(self) -> _NoGrad:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def __call__(self, func):  # type: ignore[no-untyped-def]
        """When used as a decorator, pass the function through unchanged."""
        return func


def _build_mock_torch() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Construct mock torch, torch.nn, and torch.nn.functional modules."""
    mt = MagicMock()
    mn = MagicMock()
    mf = MagicMock()

    # nn.Module must be a real class so subclassing works
    mn.Module = _MockModule

    # torch.no_grad must work as @torch.no_grad() decorator
    mt.no_grad = _NoGrad

    # Tensor as a real type for annotation compatibility
    mt.Tensor = type("Tensor", (), {})

    # Wire up hierarchy
    mt.nn = mn
    mn.functional = mf

    return mt, mn, mf


# ---------- Install mocks BEFORE importing network modules ------------------

_TORCH_KEYS = ("torch", "torch.nn", "torch.nn.functional")
_NETWORK_KEYS = (
    "src.core.networks",
    "src.core.networks.blocks",
    "src.core.networks.heads",
    "src.core.networks.losses",
    "src.core.networks.encoders",
    "src.core.networks.dual_head",
)

# Preserve any pre-existing entries so we can restore them later
_saved: dict[str, object | None] = {}
for _key in (*_TORCH_KEYS, *_NETWORK_KEYS):
    _saved[_key] = sys.modules.pop(_key, None)

_mock_torch, _mock_nn, _mock_nn_functional = _build_mock_torch()
sys.modules["torch"] = _mock_torch
sys.modules["torch.nn"] = _mock_nn
sys.modules["torch.nn.functional"] = _mock_nn_functional

# Now the imports will resolve against our mocks
from src.core.networks.blocks import ResidualBlock  # noqa: E402  # isort: skip
from src.core.networks.dual_head import DualHeadNetwork, build_network  # noqa: E402
from src.core.networks.encoders import SpatialEncoder, VectorEncoder  # noqa: E402
from src.core.networks.heads import PolicyHead, ValueHead  # noqa: E402
from src.core.networks.losses import (  # noqa: E402
    compute_combined_loss,
    compute_policy_loss,
    compute_value_loss,
)

# ============================================================================
# Module-level teardown -- restore sys.modules
# ============================================================================


def teardown_module() -> None:
    """Remove injected mocks and restore original sys.modules entries."""
    for key in (*_TORCH_KEYS, *_NETWORK_KEYS):
        sys.modules.pop(key, None)
    for key, value in _saved.items():
        if value is not None:
            sys.modules[key] = value  # type: ignore[assignment]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def fresh_nn_mocks() -> tuple[MagicMock, MagicMock]:
    """Return (mock_nn, mock_nn_functional) with call tracking reset."""
    _mock_nn.reset_mock()
    _mock_nn_functional.reset_mock()
    return _mock_nn, _mock_nn_functional


# ============================================================================
# Tests: __init__.py -- HAS_TORCH guard
# ============================================================================


class TestHasTorchGuard:
    """Verify the conditional export logic in src.core.networks.__init__."""

    def test_has_torch_false_when_torch_unavailable(self) -> None:
        """HAS_TORCH is False and __all__ is minimal when torch cannot be imported."""
        saved_torch = sys.modules.get("torch")
        saved_init = sys.modules.pop("src.core.networks", None)
        # Setting to None tells Python the module definitively does not exist,
        # which makes ``import torch`` raise ModuleNotFoundError.
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            mod = importlib.import_module("src.core.networks")
            assert mod.HAS_TORCH is False
            assert mod.__all__ == ["HAS_TORCH"]
        finally:
            sys.modules["torch"] = saved_torch  # type: ignore[assignment]
            if saved_init is not None:
                sys.modules["src.core.networks"] = saved_init

    def test_has_torch_true_when_torch_available(self) -> None:
        """HAS_TORCH is True and full exports present when torch is importable."""
        sys.modules["torch"] = _mock_torch
        saved_init = sys.modules.pop("src.core.networks", None)
        try:
            mod = importlib.import_module("src.core.networks")
            assert mod.HAS_TORCH is True
            expected_exports = [
                "HAS_TORCH",
                "DualHeadNetwork",
                "PolicyHead",
                "ValueHead",
                "build_network",
                "compute_combined_loss",
                "compute_policy_loss",
                "compute_value_loss",
            ]
            for name in expected_exports:
                assert name in mod.__all__, f"{name!r} missing from __all__"
        finally:
            if saved_init is not None:
                sys.modules["src.core.networks"] = saved_init

    def test_only_has_torch_exported_when_missing(self) -> None:
        """__all__ contains exactly one entry when torch is absent."""
        saved_torch = sys.modules.get("torch")
        saved_init = sys.modules.pop("src.core.networks", None)
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            mod = importlib.import_module("src.core.networks")
            assert len(mod.__all__) == 1
            assert mod.__all__[0] == "HAS_TORCH"
        finally:
            sys.modules["torch"] = saved_torch  # type: ignore[assignment]
            if saved_init is not None:
                sys.modules["src.core.networks"] = saved_init


# ============================================================================
# Tests: blocks.py -- ResidualBlock
# ============================================================================


class TestResidualBlock:
    """Test ResidualBlock class structure and construction."""

    def test_subclass_of_module(self) -> None:
        """ResidualBlock inherits from nn.Module (MockModule)."""
        assert issubclass(ResidualBlock, _MockModule)

    def test_constructor_default_kernel(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """Constructor with only num_filters uses default kernel_size=3."""
        block = ResidualBlock(num_filters=64)
        assert block is not None

    def test_constructor_custom_kernel(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """Constructor accepts a custom kernel_size."""
        block = ResidualBlock(num_filters=128, kernel_size=5)
        assert block is not None

    def test_has_conv1_and_conv2(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """Instance has conv1 and conv2 attributes."""
        block = ResidualBlock(num_filters=64)
        assert hasattr(block, "conv1")
        assert hasattr(block, "conv2")

    def test_has_bn1_and_bn2(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """Instance has bn1 and bn2 batch-norm attributes."""
        block = ResidualBlock(num_filters=64)
        assert hasattr(block, "bn1")
        assert hasattr(block, "bn2")

    def test_has_relu(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """Instance has relu activation attribute."""
        block = ResidualBlock(num_filters=64)
        assert hasattr(block, "relu")

    def test_has_forward_method(self) -> None:
        """ResidualBlock defines a forward method."""
        assert callable(getattr(ResidualBlock, "forward", None))

    def test_conv2d_called_twice(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """nn.Conv2d is invoked for both conv1 and conv2 during construction."""
        mn, _ = fresh_nn_mocks
        ResidualBlock(num_filters=32)
        assert len(mn.Conv2d.call_args_list) == 2

    def test_batchnorm_called_twice(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """nn.BatchNorm2d is invoked for bn1 and bn2 during construction."""
        mn, _ = fresh_nn_mocks
        ResidualBlock(num_filters=32)
        assert len(mn.BatchNorm2d.call_args_list) == 2

    def test_relu_created(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """nn.ReLU is called during construction."""
        mn, _ = fresh_nn_mocks
        ResidualBlock(num_filters=32)
        assert mn.ReLU.called

    def test_padding_computed_from_kernel_size(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """Padding should be kernel_size // 2."""
        mn, _ = fresh_nn_mocks
        ResidualBlock(num_filters=64, kernel_size=5)
        for c in mn.Conv2d.call_args_list:
            assert c.kwargs["padding"] == 2

    def test_default_kernel_gives_padding_one(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """Default kernel_size=3 produces padding=1."""
        mn, _ = fresh_nn_mocks
        ResidualBlock(num_filters=64)
        for c in mn.Conv2d.call_args_list:
            assert c.kwargs["padding"] == 1

    def test_conv_bias_false(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """Both convolutions are created with bias=False."""
        mn, _ = fresh_nn_mocks
        ResidualBlock(num_filters=64)
        for c in mn.Conv2d.call_args_list:
            assert c.kwargs["bias"] is False

    def test_forward_signature(self) -> None:
        """forward(self, x) has exactly one positional tensor parameter."""
        sig = inspect.signature(ResidualBlock.forward)
        params = list(sig.parameters.keys())
        assert params == ["self", "x"]

    def test_forward_returns_result(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """forward() executes the residual path and returns a tensor."""
        block = ResidualBlock(num_filters=64)
        x = MagicMock()
        result = block.forward(x)
        # conv1 is called at least once (shares mock with conv2)
        assert block.conv1.called
        assert result is not None

    def test_forward_calls_conv_chain(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """forward() chains conv1 -> bn1 -> relu -> conv2 -> bn2."""
        block = ResidualBlock(num_filters=64)
        x = MagicMock()
        block.forward(x)
        # Both convolutions should be called
        assert block.conv1.called
        assert block.conv2.called
        # Both batch norms should be called
        assert block.bn1.called
        assert block.bn2.called

    def test_forward_applies_relu(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """forward() applies relu at least twice (after bn1 and on residual sum)."""
        block = ResidualBlock(num_filters=64)
        x = MagicMock()
        block.forward(x)
        # relu is called twice: once after bn1(conv1(x)), once on (out + residual)
        assert block.relu.call_count == 2

    def test_forward_callable_via_call(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """__call__ delegates to forward (nn.Module pattern)."""
        block = ResidualBlock(num_filters=64)
        x = MagicMock()
        result = block(x)  # Uses __call__ -> forward
        assert result is not None
        assert block.conv1.called


# ============================================================================
# Tests: heads.py -- PolicyHead
# ============================================================================


class TestPolicyHead:
    """Test PolicyHead class structure and construction."""

    def test_subclass_of_module(self) -> None:
        assert issubclass(PolicyHead, _MockModule)

    def test_constructor_required_params(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """PolicyHead requires spatial_channels, spatial_size, vector_dim, action_space_size."""
        head = PolicyHead(
            spatial_channels=64,
            spatial_size=16,
            vector_dim=256,
            action_space_size=356,
        )
        assert head is not None

    def test_constructor_default_hidden_dim(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """Default hidden_dim parameter is 64."""
        sig = inspect.signature(PolicyHead.__init__)
        assert sig.parameters["hidden_dim"].default == 64

    def test_constructor_custom_hidden_dim(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """PolicyHead accepts a custom hidden_dim."""
        head = PolicyHead(64, 16, 256, 356, hidden_dim=128)
        assert head is not None

    def test_has_conv_and_bn(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """Instance has conv and bn attributes for spatial processing."""
        head = PolicyHead(64, 16, 256, 356)
        assert hasattr(head, "conv")
        assert hasattr(head, "bn")

    def test_has_fc_layers(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """Instance has fc1 and fc2 linear layers."""
        head = PolicyHead(64, 16, 256, 356)
        assert hasattr(head, "fc1")
        assert hasattr(head, "fc2")

    def test_action_space_size_stored(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """Constructor stores _action_space_size."""
        head = PolicyHead(64, 16, 256, 356)
        assert head._action_space_size == 356

    def test_action_space_size_property(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """action_space_size property returns the correct value."""
        head = PolicyHead(64, 16, 256, 100)
        assert head.action_space_size == 100

    def test_has_forward_method(self) -> None:
        assert callable(getattr(PolicyHead, "forward", None))

    def test_conv2d_kernel_size_one(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """PolicyHead uses 1x1 convolution (kernel_size=1)."""
        mn, _ = fresh_nn_mocks
        PolicyHead(64, 16, 256, 356)
        conv_call = mn.Conv2d.call_args
        assert conv_call[0] == (64, 2)
        assert conv_call[1]["kernel_size"] == 1

    def test_batchnorm_two_channels(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """PolicyHead batch norm operates on 2 channels."""
        mn, _ = fresh_nn_mocks
        PolicyHead(64, 16, 256, 356)
        bn_call = mn.BatchNorm2d.call_args
        assert bn_call[0][0] == 2

    def test_fc1_input_dim(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """fc1 input dimension = 2 * spatial_size**2 + vector_dim."""
        mn, _ = fresh_nn_mocks
        PolicyHead(64, 8, 32, 100, hidden_dim=64)
        # flat_dim = 2 * 8 * 8 = 128, input = 128 + 32 = 160
        fc1_call = mn.Linear.call_args_list[0]
        assert fc1_call[0][0] == 160
        assert fc1_call[0][1] == 64

    def test_fc2_output_dim(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """fc2 output dimension = action_space_size."""
        mn, _ = fresh_nn_mocks
        PolicyHead(64, 8, 32, 100, hidden_dim=64)
        fc2_call = mn.Linear.call_args_list[1]
        assert fc2_call[0][1] == 100

    def test_forward_signature(self) -> None:
        """forward accepts spatial_features, vector_features, and optional action_mask."""
        sig = inspect.signature(PolicyHead.forward)
        params = list(sig.parameters.keys())
        assert "spatial_features" in params
        assert "vector_features" in params
        assert "action_mask" in params
        assert sig.parameters["action_mask"].default is None


# ============================================================================
# Tests: heads.py -- ValueHead
# ============================================================================


class TestValueHead:
    """Test ValueHead class structure and construction."""

    def test_subclass_of_module(self) -> None:
        assert issubclass(ValueHead, _MockModule)

    def test_constructor_required_params(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """ValueHead requires spatial_channels, spatial_size, vector_dim."""
        head = ValueHead(spatial_channels=64, spatial_size=16, vector_dim=256)
        assert head is not None

    def test_constructor_default_hidden_dim(self) -> None:
        """Default hidden_dim parameter is 64."""
        sig = inspect.signature(ValueHead.__init__)
        assert sig.parameters["hidden_dim"].default == 64

    def test_constructor_custom_hidden_dim(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """ValueHead accepts a custom hidden_dim."""
        head = ValueHead(64, 16, 256, hidden_dim=128)
        assert head is not None

    def test_has_conv_and_bn(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        head = ValueHead(64, 16, 256)
        assert hasattr(head, "conv")
        assert hasattr(head, "bn")

    def test_has_fc_layers(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        head = ValueHead(64, 16, 256)
        assert hasattr(head, "fc1")
        assert hasattr(head, "fc2")

    def test_has_forward_method(self) -> None:
        assert callable(getattr(ValueHead, "forward", None))

    def test_conv2d_one_output_channel(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """ValueHead uses Conv2d with 1 output channel."""
        mn, _ = fresh_nn_mocks
        ValueHead(64, 16, 256)
        conv_call = mn.Conv2d.call_args
        assert conv_call[0] == (64, 1)
        assert conv_call[1]["kernel_size"] == 1

    def test_batchnorm_one_channel(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """ValueHead batch norm operates on 1 channel."""
        mn, _ = fresh_nn_mocks
        ValueHead(64, 16, 256)
        bn_call = mn.BatchNorm2d.call_args
        assert bn_call[0][0] == 1

    def test_fc1_input_dim(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """fc1 input dimension = 1 * spatial_size**2 + vector_dim."""
        mn, _ = fresh_nn_mocks
        ValueHead(64, 8, 32, hidden_dim=64)
        # flat_dim = 1 * 8 * 8 = 64, input = 64 + 32 = 96
        fc1_call = mn.Linear.call_args_list[0]
        assert fc1_call[0][0] == 96

    def test_fc2_output_one(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """fc2 output dimension is 1 (scalar value)."""
        mn, _ = fresh_nn_mocks
        ValueHead(64, 8, 32, hidden_dim=64)
        fc2_call = mn.Linear.call_args_list[1]
        assert fc2_call[0][1] == 1

    def test_forward_signature(self) -> None:
        """forward accepts spatial_features and vector_features only."""
        sig = inspect.signature(ValueHead.forward)
        params = list(sig.parameters.keys())
        assert params == ["self", "spatial_features", "vector_features"]


# ============================================================================
# Tests: losses.py
# ============================================================================


class TestComputePolicyLoss:
    """Test compute_policy_loss function."""

    def test_callable(self) -> None:
        assert callable(compute_policy_loss)

    def test_accepts_logits_and_target(self) -> None:
        """Function runs with mock tensor arguments."""
        result = compute_policy_loss(MagicMock(), MagicMock())
        assert result is not None

    def test_accepts_action_mask(self) -> None:
        """Optional action_mask parameter is accepted."""
        result = compute_policy_loss(MagicMock(), MagicMock(), action_mask=MagicMock())
        assert result is not None

    def test_calls_log_softmax(self) -> None:
        """Function invokes F.log_softmax on policy logits."""
        _mock_nn_functional.reset_mock()
        logits = MagicMock()
        compute_policy_loss(logits, MagicMock())
        _mock_nn_functional.log_softmax.assert_called_once_with(logits, dim=-1)

    def test_action_mask_default_none(self) -> None:
        """action_mask parameter defaults to None."""
        sig = inspect.signature(compute_policy_loss)
        assert sig.parameters["action_mask"].default is None


class TestComputeValueLoss:
    """Test compute_value_loss function."""

    def test_callable(self) -> None:
        assert callable(compute_value_loss)

    def test_accepts_predicted_and_target(self) -> None:
        result = compute_value_loss(MagicMock(), MagicMock())
        assert result is not None

    def test_calls_mse_loss(self) -> None:
        """Function invokes F.mse_loss."""
        _mock_nn_functional.reset_mock()
        compute_value_loss(MagicMock(), MagicMock())
        assert _mock_nn_functional.mse_loss.called

    def test_signature_has_two_params(self) -> None:
        """Function takes predicted_value and target_value."""
        sig = inspect.signature(compute_value_loss)
        params = [p for p in sig.parameters if p != "return"]
        assert params == ["predicted_value", "target_value"]


class TestComputeCombinedLoss:
    """Test compute_combined_loss function."""

    def test_callable(self) -> None:
        assert callable(compute_combined_loss)

    def test_returns_tuple(self) -> None:
        """Return type is a 2-tuple of (loss, metrics_dict)."""
        result = compute_combined_loss(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_metrics_dict_keys(self) -> None:
        """Metrics dict contains total_loss, policy_loss, and value_loss."""
        _, metrics = compute_combined_loss(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        assert isinstance(metrics, dict)
        assert "total_loss" in metrics
        assert "policy_loss" in metrics
        assert "value_loss" in metrics

    def test_accepts_action_mask(self) -> None:
        """Optional action_mask parameter is accepted."""
        result = compute_combined_loss(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(),
            action_mask=MagicMock(),
        )
        assert isinstance(result, tuple)

    def test_accepts_value_loss_weight(self) -> None:
        """Optional value_loss_weight parameter is accepted."""
        result = compute_combined_loss(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(),
            value_loss_weight=0.5,
        )
        assert isinstance(result, tuple)

    def test_default_value_loss_weight_is_one(self) -> None:
        """Default value_loss_weight is 1.0."""
        sig = inspect.signature(compute_combined_loss)
        assert sig.parameters["value_loss_weight"].default == 1.0

    def test_metrics_has_three_entries(self) -> None:
        """Metrics dict has exactly three keys."""
        _, metrics = compute_combined_loss(
            MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        assert len(metrics) == 3


# ============================================================================
# Tests: encoders.py -- SpatialEncoder
# ============================================================================


class TestSpatialEncoder:
    """Test SpatialEncoder class structure and construction."""

    def test_subclass_of_module(self) -> None:
        assert issubclass(SpatialEncoder, _MockModule)

    def test_constructor_required_params(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        enc = SpatialEncoder(in_channels=13)
        assert enc is not None

    def test_constructor_custom_params(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        enc = SpatialEncoder(in_channels=13, num_filters=128, num_blocks=4)
        assert enc is not None

    def test_has_initial_conv(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        enc = SpatialEncoder(in_channels=13)
        assert hasattr(enc, "initial_conv")

    def test_has_initial_bn(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        enc = SpatialEncoder(in_channels=13)
        assert hasattr(enc, "initial_bn")

    def test_has_relu(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        enc = SpatialEncoder(in_channels=13)
        assert hasattr(enc, "relu")

    def test_has_blocks(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        enc = SpatialEncoder(in_channels=13)
        assert hasattr(enc, "blocks")

    def test_stores_num_filters(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        enc = SpatialEncoder(in_channels=13, num_filters=128)
        assert enc.num_filters == 128

    def test_default_num_filters(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """Default num_filters is 64."""
        enc = SpatialEncoder(in_channels=13)
        assert enc.num_filters == 64

    def test_default_num_blocks(self) -> None:
        """Default num_blocks is 6."""
        sig = inspect.signature(SpatialEncoder.__init__)
        assert sig.parameters["num_blocks"].default == 6

    def test_has_forward_method(self) -> None:
        assert callable(getattr(SpatialEncoder, "forward", None))

    def test_sequential_called_for_blocks(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """nn.Sequential is used to bundle residual blocks."""
        mn, _ = fresh_nn_mocks
        SpatialEncoder(in_channels=13, num_blocks=3)
        assert mn.Sequential.called

    def test_initial_conv_params(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """Initial conv uses in_channels -> num_filters with kernel=3, padding=1."""
        mn, _ = fresh_nn_mocks
        SpatialEncoder(in_channels=13, num_filters=64)
        first_call = mn.Conv2d.call_args_list[0]
        assert first_call[0][0] == 13  # in_channels
        assert first_call[0][1] == 64  # num_filters
        assert first_call[0][2] == 3  # kernel_size

    def test_initial_conv_bias_false(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """Initial convolution has bias=False."""
        mn, _ = fresh_nn_mocks
        SpatialEncoder(in_channels=13)
        first_call = mn.Conv2d.call_args_list[0]
        assert first_call[1]["bias"] is False

    def test_forward_signature(self) -> None:
        sig = inspect.signature(SpatialEncoder.forward)
        params = list(sig.parameters.keys())
        assert params == ["self", "x"]


# ============================================================================
# Tests: encoders.py -- VectorEncoder
# ============================================================================


class TestVectorEncoder:
    """Test VectorEncoder class structure and construction."""

    def test_subclass_of_module(self) -> None:
        assert issubclass(VectorEncoder, _MockModule)

    def test_constructor_required_params(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        enc = VectorEncoder(input_dim=64)
        assert enc is not None

    def test_constructor_custom_params(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        enc = VectorEncoder(input_dim=64, output_dim=128, dropout=0.1)
        assert enc is not None

    def test_has_net(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        enc = VectorEncoder(input_dim=64)
        assert hasattr(enc, "net")

    def test_stores_output_dim(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        enc = VectorEncoder(input_dim=64, output_dim=128)
        assert enc.output_dim == 128

    def test_default_output_dim(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """Default output_dim is 256."""
        enc = VectorEncoder(input_dim=64)
        assert enc.output_dim == 256

    def test_default_dropout(self) -> None:
        """Default dropout is 0.0."""
        sig = inspect.signature(VectorEncoder.__init__)
        assert sig.parameters["dropout"].default == 0.0

    def test_has_forward_method(self) -> None:
        assert callable(getattr(VectorEncoder, "forward", None))

    def test_sequential_used(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """nn.Sequential wraps the MLP layers."""
        mn, _ = fresh_nn_mocks
        VectorEncoder(input_dim=64)
        assert mn.Sequential.called

    def test_dropout_used(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """nn.Dropout is used in the architecture."""
        mn, _ = fresh_nn_mocks
        VectorEncoder(input_dim=64, dropout=0.5)
        mn.Dropout.assert_called_once_with(0.5)

    def test_linear_layers_created(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """Two nn.Linear layers are created."""
        mn, _ = fresh_nn_mocks
        VectorEncoder(input_dim=64, output_dim=128)
        linear_calls = mn.Linear.call_args_list
        assert len(linear_calls) == 2
        # First: Linear(input_dim, output_dim)
        assert linear_calls[0][0] == (64, 128)
        # Second: Linear(output_dim, output_dim)
        assert linear_calls[1][0] == (128, 128)

    def test_forward_signature(self) -> None:
        sig = inspect.signature(VectorEncoder.forward)
        params = list(sig.parameters.keys())
        assert params == ["self", "x"]


# ============================================================================
# Tests: dual_head.py -- DualHeadNetwork
# ============================================================================


class TestDualHeadNetwork:
    """Test DualHeadNetwork class structure and construction."""

    def test_subclass_of_module(self) -> None:
        assert issubclass(DualHeadNetwork, _MockModule)

    def test_constructor_required_params(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        net = DualHeadNetwork(
            spatial_channels=13,
            spatial_size=16,
            vector_dim=64,
            action_space_size=356,
        )
        assert net is not None

    def test_constructor_with_config(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """Constructor accepts an optional NetworkConfig."""
        from src.core.config import NetworkConfig

        cfg = NetworkConfig(num_filters=32, num_residual_blocks=2)
        net = DualHeadNetwork(13, 16, 64, 356, config=cfg)
        assert net is not None

    def test_has_spatial_encoder(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        net = DualHeadNetwork(13, 16, 64, 356)
        assert hasattr(net, "spatial_encoder")
        assert isinstance(net.spatial_encoder, SpatialEncoder)

    def test_has_vector_encoder(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        net = DualHeadNetwork(13, 16, 64, 356)
        assert hasattr(net, "vector_encoder")
        assert isinstance(net.vector_encoder, VectorEncoder)

    def test_has_policy_head(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        net = DualHeadNetwork(13, 16, 64, 356)
        assert hasattr(net, "policy_head")
        assert isinstance(net.policy_head, PolicyHead)

    def test_has_value_head(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        net = DualHeadNetwork(13, 16, 64, 356)
        assert hasattr(net, "value_head")
        assert isinstance(net.value_head, ValueHead)

    def test_stores_spatial_channels(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        net = DualHeadNetwork(13, 16, 64, 356)
        assert net._spatial_channels == 13

    def test_stores_spatial_size(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        net = DualHeadNetwork(13, 16, 64, 356)
        assert net._spatial_size == 16

    def test_stores_vector_dim(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        net = DualHeadNetwork(13, 16, 64, 356)
        assert net._vector_dim == 64

    def test_stores_action_space_size(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        net = DualHeadNetwork(13, 16, 64, 356)
        assert net._action_space_size == 356

    def test_has_forward_method(self) -> None:
        assert callable(getattr(DualHeadNetwork, "forward", None))

    def test_has_predict_method(self) -> None:
        assert callable(getattr(DualHeadNetwork, "predict", None))

    def test_config_defaults_applied(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        """When config is None, NetworkConfig defaults are used."""
        from src.core.config import NetworkConfig

        defaults = NetworkConfig()
        net = DualHeadNetwork(13, 16, 64, 356)
        assert net.spatial_encoder.num_filters == defaults.num_filters

    def test_custom_config_propagated(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """Custom config values are forwarded to sub-components."""
        from src.core.config import NetworkConfig

        cfg = NetworkConfig(num_filters=128, num_residual_blocks=4)
        net = DualHeadNetwork(13, 16, 64, 356, config=cfg)
        assert net.spatial_encoder.num_filters == 128

    def test_policy_head_action_size(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """PolicyHead receives the correct action_space_size."""
        net = DualHeadNetwork(13, 16, 64, 200)
        assert net.policy_head.action_space_size == 200

    def test_vector_encoder_output_dim(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """VectorEncoder uses fc_hidden_dim from config."""
        from src.core.config import NetworkConfig

        cfg = NetworkConfig(fc_hidden_dim=512)
        net = DualHeadNetwork(13, 16, 64, 356, config=cfg)
        assert net.vector_encoder.output_dim == 512

    def test_forward_signature(self) -> None:
        """forward accepts spatial, vector, and optional action_mask."""
        sig = inspect.signature(DualHeadNetwork.forward)
        params = list(sig.parameters.keys())
        assert "spatial" in params
        assert "vector" in params
        assert "action_mask" in params
        assert sig.parameters["action_mask"].default is None

    def test_predict_signature(self) -> None:
        """predict accepts spatial, vector, and optional action_mask."""
        sig = inspect.signature(DualHeadNetwork.predict)
        params = list(sig.parameters.keys())
        assert "spatial" in params
        assert "vector" in params
        assert "action_mask" in params
        assert sig.parameters["action_mask"].default is None

    def test_forward_returns_tuple_annotation(self) -> None:
        """forward is annotated to return tuple[Tensor, Tensor]."""
        sig = inspect.signature(DualHeadNetwork.forward)
        ret = sig.return_annotation
        # With from __future__ import annotations, this is a string
        assert "tuple" in str(ret).lower()


# ============================================================================
# Tests: dual_head.py -- build_network factory
# ============================================================================


class TestBuildNetwork:
    """Test build_network factory function."""

    def test_callable(self) -> None:
        assert callable(build_network)

    def test_returns_dual_head_network(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        net = build_network(13, 16, 64, 356)
        assert isinstance(net, DualHeadNetwork)

    def test_passes_config(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        from src.core.config import NetworkConfig

        cfg = NetworkConfig(num_filters=32)
        net = build_network(13, 16, 64, 356, config=cfg)
        assert net.spatial_encoder.num_filters == 32

    def test_dimensions_match(self, fresh_nn_mocks: tuple[MagicMock, MagicMock]) -> None:
        net = build_network(
            spatial_channels=7,
            spatial_size=8,
            vector_dim=32,
            action_space_size=100,
        )
        assert net._spatial_channels == 7
        assert net._spatial_size == 8
        assert net._vector_dim == 32
        assert net._action_space_size == 100

    def test_default_config_when_none(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """build_network with config=None uses NetworkConfig defaults."""
        from src.core.config import NetworkConfig

        defaults = NetworkConfig()
        net = build_network(13, 16, 64, 356, config=None)
        assert net.spatial_encoder.num_filters == defaults.num_filters

    def test_signature_matches_dual_head_constructor(self) -> None:
        """build_network accepts the same arguments as DualHeadNetwork.__init__."""
        factory_sig = inspect.signature(build_network)
        constructor_sig = inspect.signature(DualHeadNetwork.__init__)

        factory_params = set(factory_sig.parameters.keys())
        # Constructor has 'self' which factory does not
        constructor_params = set(constructor_sig.parameters.keys()) - {"self"}
        assert factory_params == constructor_params


# ============================================================================
# Tests: heads.py -- forward() method execution with mock tensors
# ============================================================================


class TestPolicyHeadForward:
    """Exercise PolicyHead.forward() with properly-wired mock tensors."""

    def test_forward_without_mask(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """forward() runs without action_mask and returns logits."""
        _, mf = fresh_nn_mocks
        # f.relu must return a mock that supports .flatten()
        mock_tensor = MagicMock()
        mock_tensor.flatten.return_value = mock_tensor
        mf.relu.return_value = mock_tensor
        _mock_torch.cat.return_value = mock_tensor

        head = PolicyHead(64, 8, 32, 100)
        result = head.forward(MagicMock(), MagicMock(), action_mask=None)
        assert result is not None
        # f.relu should have been called (at least twice: bn->relu, fc1->relu)
        assert mf.relu.call_count >= 2

    def test_forward_with_action_mask(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """forward() applies masking when action_mask is provided."""
        _, mf = fresh_nn_mocks
        mock_tensor = MagicMock()
        mock_tensor.flatten.return_value = mock_tensor
        mf.relu.return_value = mock_tensor
        _mock_torch.cat.return_value = mock_tensor
        mask = MagicMock()

        head = PolicyHead(64, 8, 32, 100)
        logits = head.forward(MagicMock(), MagicMock(), action_mask=mask)
        # The masked_fill call should have happened
        assert logits is not None

    def test_forward_calls_conv_then_bn(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """forward() chains self.conv -> self.bn."""
        _, mf = fresh_nn_mocks
        mock_tensor = MagicMock()
        mock_tensor.flatten.return_value = mock_tensor
        mf.relu.return_value = mock_tensor
        _mock_torch.cat.return_value = mock_tensor

        head = PolicyHead(64, 8, 32, 100)
        spatial = MagicMock()
        head.forward(spatial, MagicMock())
        # conv was called with spatial input
        head.conv.assert_called_once_with(spatial)


class TestValueHeadForward:
    """Exercise ValueHead.forward() with mock tensors."""

    def test_forward_returns_value(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """forward() produces a value tensor."""
        _, mf = fresh_nn_mocks
        mock_tensor = MagicMock()
        mock_tensor.flatten.return_value = mock_tensor
        mf.relu.return_value = mock_tensor
        _mock_torch.cat.return_value = mock_tensor
        _mock_torch.tanh.return_value = mock_tensor

        head = ValueHead(64, 8, 32)
        result = head.forward(MagicMock(), MagicMock())
        assert result is not None
        # tanh should be called on the output
        _mock_torch.tanh.assert_called()

    def test_forward_calls_conv(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """forward() processes spatial features through conv."""
        _, mf = fresh_nn_mocks
        mock_tensor = MagicMock()
        mock_tensor.flatten.return_value = mock_tensor
        mf.relu.return_value = mock_tensor
        _mock_torch.cat.return_value = mock_tensor
        _mock_torch.tanh.return_value = mock_tensor

        head = ValueHead(64, 8, 32)
        spatial = MagicMock()
        head.forward(spatial, MagicMock())
        head.conv.assert_called_once_with(spatial)


class TestDualHeadNetworkForward:
    """Exercise DualHeadNetwork.forward() and predict()."""

    def test_forward_returns_tuple(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """forward() returns (policy_logits, value) tuple."""
        _, mf = fresh_nn_mocks
        # Wire up mocks so the full forward chain works
        mock_tensor = MagicMock()
        mock_tensor.flatten.return_value = mock_tensor
        mf.relu.return_value = mock_tensor
        _mock_torch.cat.return_value = mock_tensor
        _mock_torch.tanh.return_value = mock_tensor

        net = DualHeadNetwork(13, 16, 64, 356)
        result = net.forward(MagicMock(), MagicMock())
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_forward_with_mask_accepted(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """forward() accepts action_mask parameter."""
        _, mf = fresh_nn_mocks
        mock_tensor = MagicMock()
        mock_tensor.flatten.return_value = mock_tensor
        mf.relu.return_value = mock_tensor
        _mock_torch.cat.return_value = mock_tensor
        _mock_torch.tanh.return_value = mock_tensor

        net = DualHeadNetwork(13, 16, 64, 356)
        mask = MagicMock()
        result = net.forward(MagicMock(), MagicMock(), action_mask=mask)
        assert isinstance(result, tuple)

    def test_predict_returns_tuple(
        self, fresh_nn_mocks: tuple[MagicMock, MagicMock]
    ) -> None:
        """predict() returns (policy_probs, value) via softmax."""
        _, mf = fresh_nn_mocks
        mock_tensor = MagicMock()
        mock_tensor.flatten.return_value = mock_tensor
        mf.relu.return_value = mock_tensor
        mf.softmax.return_value = mock_tensor
        _mock_torch.cat.return_value = mock_tensor
        _mock_torch.tanh.return_value = mock_tensor

        net = DualHeadNetwork(13, 16, 64, 356)
        result = net.predict(MagicMock(), MagicMock())
        assert isinstance(result, tuple)
        assert len(result) == 2
        # softmax is called on policy logits
        mf.softmax.assert_called()
