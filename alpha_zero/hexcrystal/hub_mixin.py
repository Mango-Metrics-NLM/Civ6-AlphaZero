"""Hugging Face Hub integration for the hex-crystalline spatial encoder.

This module is **opt-in**. Importing it requires the ``hub`` extra::

    pip install -e '.[hub]'

Keeping the integration in its own module is deliberate. The rest of
:mod:`alpha_zero.hexcrystal` depends on torch and the standard library alone, and
that contract is what makes the encoder reusable elsewhere. Adding
``PyTorchModelHubMixin`` as a conditional base class on
:class:`~alpha_zero.hexcrystal.hex_encoder.HexSpatialEncoder` itself would have
meant a ``try``/``except ImportError`` around a class definition, which type
checkers reject (inconsistent MRO, "variable is not valid as a type") and which
would make the dependency effectively mandatory. A separate module sidesteps the
conditional entirely: import it and you need the extra, ignore it and the base
encoder is untouched.

Why the constructor takes flat keyword arguments rather than a config object:
the mixin captures ``__init__`` arguments to produce ``config.json``, so flat
primitives yield a flat, human-readable config document that round-trips through
:meth:`HexEncoderConfig.from_dict` without custom encoders.

Example:
    >>> from alpha_zero.hexcrystal.hub_mixin import HexSpatialEncoderForHub
    >>> encoder = HexSpatialEncoderForHub(in_channels=13, spatial_size=8)
    >>> encoder.save_pretrained("./my-encoder")           # doctest: +SKIP
    >>> reloaded = HexSpatialEncoderForHub.from_pretrained("./my-encoder")  # doctest: +SKIP

Note:
    ``save_pretrained`` writes a placeholder ``README.md`` alongside the weights.
    Publishing tools should overwrite it with a real model card.
"""

from __future__ import annotations

import logging

from alpha_zero.hexcrystal.hex_encoder import HexEncoderConfig, HexSpatialEncoder

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
    raise ImportError(
        "alpha_zero.hexcrystal.hub_mixin requires the 'hub' extra. "
        "Install it with:  pip install -e '.[hub]'"
    ) from exc

logger = logging.getLogger(__name__)

__all__ = ["HEX_ENCODER_HUB_TAGS", "HUB_LIBRARY_NAME", "HexSpatialEncoderForHub"]

# Hub metadata. Deliberately *not* passed as class keyword arguments to the
# mixin: those only populate the placeholder README that `save_pretrained`
# writes, which publishing tools overwrite with a real model card anyway, and
# they do not type-check against the mixin's `__init_subclass__` signature.
# Exported here so the model card and the upload tool share one source.
HUB_LIBRARY_NAME = "alpha-zero-hexcrystal"
HEX_ENCODER_HUB_TAGS = (
    "pytorch",
    "hexagonal-grid",
    "spatial-encoder",
    "reinforcement-learning",
)


class HexSpatialEncoderForHub(HexSpatialEncoder, PyTorchModelHubMixin):
    """A :class:`HexSpatialEncoder` that can be saved to and loaded from the Hub.

    Behaviourally identical to its parent — same layers, same forward pass, same
    ``state_dict`` keys — so weights transfer between the two classes in either
    direction. The only additions are ``save_pretrained`` and ``from_pretrained``,
    inherited from the mixin, which serialize weights as safetensors and the
    constructor arguments as ``config.json``.

    Args:
        in_channels: Number of input spatial channels.
        embed_dim: Embedding dimension for output tokens.
        spatial_size: Size of the (square) spatial grid.
        hex_hidden_dim: Hidden dimension of the hexagonal convolution stack.
        hex_conv_layers: Number of hexagonal convolution layers.
        lattice_type: Symmetry group — ``"hexagonal"``, ``"cubic"``, or
            ``"tetrahedral"``.
        num_output_tokens: Number of output tokens.
        dropout: Dropout probability.
        offset_type: Hex offset coordinate system. Only ``"even-q"`` is
            implemented.
        activation: Activation function name.
        use_batch_norm: Whether hexagonal convolutions use batch normalization.
        use_crystalline: Whether to apply the crystalline lattice layer.
    """

    def __init__(
        self,
        in_channels: int = 13,
        embed_dim: int = 64,
        spatial_size: int = 8,
        hex_hidden_dim: int = 32,
        hex_conv_layers: int = 2,
        lattice_type: str = "hexagonal",
        num_output_tokens: int = 4,
        dropout: float = 0.1,
        offset_type: str = "even-q",
        activation: str = "gelu",
        use_batch_norm: bool = True,
        use_crystalline: bool = True,
    ) -> None:
        config = HexEncoderConfig(
            in_channels=in_channels,
            embed_dim=embed_dim,
            spatial_size=spatial_size,
            hex_hidden_dim=hex_hidden_dim,
            hex_conv_layers=hex_conv_layers,
            lattice_type=lattice_type,
            num_output_tokens=num_output_tokens,
            dropout=dropout,
            offset_type=offset_type,
            activation=activation,
            use_batch_norm=use_batch_norm,
            use_crystalline=use_crystalline,
        )
        super().__init__(config)

        logger.debug(
            "HexSpatialEncoderForHub ready: %d params, crystalline=%s",
            sum(p.numel() for p in self.parameters()),
            use_crystalline,
        )

    @classmethod
    def from_encoder_config(cls, config: HexEncoderConfig) -> HexSpatialEncoderForHub:
        """Build a hub-enabled encoder from a :class:`HexEncoderConfig`.

        Args:
            config: Source configuration.

        Returns:
            A new hub-enabled encoder with the same settings.
        """
        return cls(**config.to_dict())

    @classmethod
    def from_encoder(cls, encoder: HexSpatialEncoder) -> HexSpatialEncoderForHub:
        """Copy a plain encoder's configuration and weights into a hub-enabled one.

        This is the bridge used when publishing: training produces a plain
        :class:`HexSpatialEncoder` (often nested inside a larger network), and
        publication needs the hub-enabled subclass. Weights transfer strictly,
        so a mismatched architecture raises rather than loading partially.

        Args:
            encoder: A trained :class:`HexSpatialEncoder`.

        Returns:
            A hub-enabled encoder carrying the same weights.

        Raises:
            RuntimeError: If the source weights do not match the rebuilt
                architecture exactly.
        """
        hub_encoder = cls.from_encoder_config(encoder.config)
        hub_encoder.load_state_dict(encoder.state_dict(), strict=True)
        return hub_encoder
