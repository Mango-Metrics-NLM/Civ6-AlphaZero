"""Hugging Face Hub integration for the MuZero world-model network.

This module is **opt-in**. Importing it requires the ``hub`` extra::

    pip install -e '.[hub]'

Same design rationale as :mod:`alpha_zero.hexcrystal.hub_mixin` and
:mod:`alpha_zero.core.transformer_hub_mixin`: a separate module keeps
``PyTorchModelHubMixin`` out of the base class's MRO.

Example::

    >>> from alpha_zero.core.muzero_hub_mixin import MuZeroNetworkForHub
    >>> model = MuZeroNetworkForHub(spatial_channels=13, spatial_size=16)
    >>> model.save_pretrained("./my-muzero")                  # doctest: +SKIP
    >>> reloaded = MuZeroNetworkForHub.from_pretrained("./my-muzero")
    ...                                                       # doctest: +SKIP
"""

from __future__ import annotations

import logging
from dataclasses import fields as dc_fields
from typing import Any

from alpha_zero.core.muzero_network import MuZeroConfig, MuZeroNetwork
from alpha_zero.hub.constants import DEFAULT_MODEL_TAGS, LIBRARY_NAME  # noqa: F401
from alpha_zero.utils.constants import DEFAULT_ACTION_SPACE_SIZE

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "alpha_zero.core.muzero_hub_mixin requires the 'hub' extra. "
        "Install with:  pip install -e '.[hub]'"
    ) from exc

logger = logging.getLogger(__name__)

__all__ = [
    "MUZERO_HUB_TAGS",
    "MuZeroNetworkForHub",
]

#: Tags specific to the MuZero world-model network.
MUZERO_HUB_TAGS: tuple[str, ...] = (
    *DEFAULT_MODEL_TAGS,
    "muzero",
    "world-model",
    "latent-dynamics",
)


def _config_to_flat_kwargs(config: MuZeroConfig) -> dict[str, Any]:
    """Extract serialisable kwargs from a MuZeroConfig."""
    return {f.name: getattr(config, f.name) for f in dc_fields(config)}


class MuZeroNetworkForHub(MuZeroNetwork, PyTorchModelHubMixin):
    """A :class:`MuZeroNetwork` that can be saved to / loaded from the Hub.

    Behaviourally identical to its parent — same layers, same forward pass,
    same ``state_dict`` keys.

    Args:
        spatial_channels: Number of input spatial channels.
        spatial_size: Spatial grid size (H=W).
        vector_dim: Vector input dimension.
        num_res_blocks: Residual blocks in representation function.
        num_filters: Filters in representation function.
        fc_hidden_dim: Fully connected hidden dimension.
        action_space_size: Total number of actions.
        value_head_hidden: Value head hidden dimension.
        dropout: Dropout probability.
        hidden_state_channels: Channels in latent hidden state.
        dynamics_num_blocks: ResBlocks in dynamics function.
        action_embedding_dim: Action embedding dimension.
        reward_support_size: Categorical reward range.
        use_categorical_reward: Use categorical reward prediction.
        use_consistency_loss: EfficientZero consistency loss.
        projection_dim: Projection head dimension.
        prediction_hidden_dim: Prediction head hidden dimension.
        use_value_prefix: Value prefix prediction.
        value_prefix_horizon: Prefix prediction horizon.
        unroll_steps: Dynamics unroll steps.
    """

    def __init__(
        self,
        spatial_channels: int = 13,
        spatial_size: int = 16,
        vector_dim: int = 64,
        num_res_blocks: int = 6,
        num_filters: int = 64,
        fc_hidden_dim: int = 256,
        action_space_size: int = DEFAULT_ACTION_SPACE_SIZE,
        value_head_hidden: int = 128,
        dropout: float = 0.0,
        hidden_state_channels: int = 64,
        dynamics_num_blocks: int = 4,
        action_embedding_dim: int = 32,
        reward_support_size: int = 601,
        use_categorical_reward: bool = True,
        use_consistency_loss: bool = True,
        projection_dim: int = 256,
        prediction_hidden_dim: int = 128,
        use_value_prefix: bool = True,
        value_prefix_horizon: int = 5,
        unroll_steps: int = 5,
    ) -> None:
        config = MuZeroConfig(
            spatial_channels=spatial_channels,
            spatial_size=spatial_size,
            vector_dim=vector_dim,
            num_res_blocks=num_res_blocks,
            num_filters=num_filters,
            fc_hidden_dim=fc_hidden_dim,
            action_space_size=action_space_size,
            value_head_hidden=value_head_hidden,
            dropout=dropout,
            hidden_state_channels=hidden_state_channels,
            dynamics_num_blocks=dynamics_num_blocks,
            action_embedding_dim=action_embedding_dim,
            reward_support_size=reward_support_size,
            use_categorical_reward=use_categorical_reward,
            use_consistency_loss=use_consistency_loss,
            projection_dim=projection_dim,
            prediction_hidden_dim=prediction_hidden_dim,
            use_value_prefix=use_value_prefix,
            value_prefix_horizon=value_prefix_horizon,
            unroll_steps=unroll_steps,
        )
        super().__init__(config)

        logger.debug(
            "MuZeroNetworkForHub ready: %d params",
            sum(p.numel() for p in self.parameters()),
        )

    @classmethod
    def from_muzero_config(cls, config: MuZeroConfig) -> MuZeroNetworkForHub:
        """Build a hub-enabled network from a :class:`MuZeroConfig`.

        Args:
            config: Source configuration.

        Returns:
            A new hub-enabled network with the same settings.
        """
        return cls(**_config_to_flat_kwargs(config))

    @classmethod
    def from_network(cls, network: MuZeroNetwork) -> MuZeroNetworkForHub:
        """Copy a plain network's config and weights into a hub-enabled one.

        Args:
            network: A trained :class:`MuZeroNetwork`.

        Returns:
            A hub-enabled network carrying the same weights.

        Raises:
            RuntimeError: If the source weights do not match the rebuilt
                architecture exactly.
        """
        hub_net = cls.from_muzero_config(network.config)
        hub_net.load_state_dict(network.state_dict(), strict=True)
        return hub_net
