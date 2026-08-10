"""Hugging Face Hub integration for the Civ6 transformer policy/value network.

This module is **opt-in**. Importing it requires the ``hub`` extra::

    pip install -e '.[hub]'

Follows the same design as :mod:`alpha_zero.hexcrystal.hub_mixin`:
a separate module keeps ``PyTorchModelHubMixin`` out of the base class's
MRO, avoiding conditional imports that break type checkers.

Example::

    >>> from alpha_zero.core.transformer_hub_mixin import Civ6TransformerNetworkForHub
    >>> model = Civ6TransformerNetworkForHub(spatial_channels=13, spatial_size=8)
    >>> model.save_pretrained("./my-transformer")            # doctest: +SKIP
    >>> reloaded = Civ6TransformerNetworkForHub.from_pretrained("./my-transformer")
    ...                                                      # doctest: +SKIP
"""

from __future__ import annotations

import logging
from dataclasses import fields as dc_fields
from typing import Any

from alpha_zero.core.transformer_network import (
    Civ6TransformerNetwork,
    TransformerConfig,
)
from alpha_zero.hub.constants import DEFAULT_MODEL_TAGS

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "alpha_zero.core.transformer_hub_mixin requires the 'hub' extra. "
        "Install with:  pip install -e '.[hub]'"
    ) from exc

logger = logging.getLogger(__name__)

__all__ = [
    "TRANSFORMER_HUB_TAGS",
    "Civ6TransformerNetworkForHub",
]

#: Tags specific to the transformer policy/value network.
TRANSFORMER_HUB_TAGS: tuple[str, ...] = (
    *DEFAULT_MODEL_TAGS,
    "transformer",
    "alphazero",
    "policy-value-network",
)


def _config_to_flat_kwargs(config: TransformerConfig) -> dict[str, Any]:
    """Extract serialisable kwargs from a TransformerConfig.

    Only includes fields that are init=True and JSON-serialisable
    (primitives and enums).  Enum values are converted to their
    ``.value`` string representation.
    """
    kwargs: dict[str, Any] = {}
    for f in dc_fields(config):
        if not f.init:
            continue
        val = getattr(config, f.name)
        # Convert enums to their string value for JSON serialisation.
        if hasattr(val, "value"):
            val = val.value
        kwargs[f.name] = val
    return kwargs


class Civ6TransformerNetworkForHub(Civ6TransformerNetwork, PyTorchModelHubMixin):
    """A :class:`Civ6TransformerNetwork` that can be saved to / loaded from the Hub.

    Behaviourally identical to its parent — same layers, same forward pass,
    same ``state_dict`` keys — so weights transfer between the two classes
    in either direction.

    The constructor takes flat keyword arguments rather than a config object
    because the mixin captures ``__init__`` arguments to produce
    ``config.json``, and flat primitives yield a human-readable config that
    round-trips without custom encoders.

    Args:
        spatial_channels: Number of input spatial channels.
        spatial_size: Size of the spatial input (H=W).
        vector_dim: Dimension of vector input features.
        action_space_size: Total number of actions.
        patch_size: Patch size for ViT embedding.
        embed_dim: Transformer embedding dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to embed_dim.
        dropout: General dropout probability.
        attention_dropout: Attention-specific dropout.
        path_dropout: Stochastic depth rate.
        norm_type: Normalization type value string.
        attention_type: Attention mechanism value string.
        use_pre_norm: Pre-LayerNorm (vs Post-LN).
        use_rope: Rotary Position Embeddings.
        use_qkv_bias: QKV projection bias.
        value_head_dim: Value head hidden dimension.
        policy_hidden_dim: Policy head hidden dimension.
        num_policy_heads: Number of policy heads.
        use_moe: Mixture of Experts.
        num_experts: Number of MoE experts.
        experts_per_token: Experts routed per token.
        cls_token: Use CLS token.
        pool_type: Feature pooling type.
        encoder_type: Encoder type selection.
    """

    def __init__(
        self,
        spatial_channels: int = 13,
        spatial_size: int = 8,
        vector_dim: int = 64,
        action_space_size: int = 376,
        patch_size: int = 4,
        embed_dim: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        path_dropout: float = 0.1,
        norm_type: str = "rms_norm",
        attention_type: str = "standard",
        use_pre_norm: bool = True,
        use_rope: bool = True,
        use_qkv_bias: bool = False,
        value_head_dim: int = 64,
        policy_hidden_dim: int = 128,
        num_policy_heads: int = 1,
        use_moe: bool = False,
        num_experts: int = 8,
        experts_per_token: int = 2,
        cls_token: bool = True,
        pool_type: str = "cls",
        encoder_type: str = "patch_embed",
    ) -> None:
        from alpha_zero.core.transformer_network import AttentionType, NormType

        config = TransformerConfig(
            spatial_channels=spatial_channels,
            spatial_size=spatial_size,
            vector_dim=vector_dim,
            action_space_size=action_space_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            path_dropout=path_dropout,
            norm_type=NormType(norm_type),
            attention_type=AttentionType(attention_type),
            use_pre_norm=use_pre_norm,
            use_rope=use_rope,
            use_qkv_bias=use_qkv_bias,
            value_head_dim=value_head_dim,
            policy_hidden_dim=policy_hidden_dim,
            num_policy_heads=num_policy_heads,
            use_moe=use_moe,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            cls_token=cls_token,
            pool_type=pool_type,
            encoder_type=encoder_type,
        )
        super().__init__(config)

        logger.debug(
            "Civ6TransformerNetworkForHub ready: %d params",
            sum(p.numel() for p in self.parameters()),
        )

    @classmethod
    def from_transformer_config(
        cls, config: TransformerConfig,
    ) -> Civ6TransformerNetworkForHub:
        """Build a hub-enabled network from a :class:`TransformerConfig`.

        Args:
            config: Source configuration.

        Returns:
            A new hub-enabled network with the same settings.
        """
        return cls(**_config_to_flat_kwargs(config))

    @classmethod
    def from_network(
        cls, network: Civ6TransformerNetwork,
    ) -> Civ6TransformerNetworkForHub:
        """Copy a plain network's config and weights into a hub-enabled one.

        This is the bridge used when publishing: training produces a plain
        :class:`Civ6TransformerNetwork`, and publication needs the hub-enabled
        subclass.  Weights transfer strictly, so a mismatched architecture
        raises rather than loading partially.

        Args:
            network: A trained :class:`Civ6TransformerNetwork`.

        Returns:
            A hub-enabled network carrying the same weights.

        Raises:
            RuntimeError: If the source weights do not match the rebuilt
                architecture exactly.
        """
        hub_net = cls.from_transformer_config(network.config)
        hub_net.load_state_dict(network.state_dict(), strict=True)
        return hub_net
