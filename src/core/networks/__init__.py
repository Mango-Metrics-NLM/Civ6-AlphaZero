"""Neural network abstractions for the Strategic Planning SDK.

Components are only available when PyTorch is installed.  Use
``HAS_TORCH`` to guard imports at call sites::

    from src.core.networks import HAS_TORCH
    if HAS_TORCH:
        from src.core.networks.dual_head import DualHeadNetwork
"""

from __future__ import annotations

try:
    import torch as _torch  # noqa: F401

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False

__all__ = ["HAS_TORCH"]

if HAS_TORCH:
    from src.core.networks.dual_head import DualHeadNetwork, build_network
    from src.core.networks.heads import PolicyHead, ValueHead
    from src.core.networks.losses import (
        compute_combined_loss,
        compute_policy_loss,
        compute_value_loss,
    )

    __all__ += [
        "DualHeadNetwork",
        "PolicyHead",
        "ValueHead",
        "build_network",
        "compute_combined_loss",
        "compute_policy_loss",
        "compute_value_loss",
    ]
