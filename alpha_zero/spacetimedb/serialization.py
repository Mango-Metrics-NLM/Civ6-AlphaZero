"""Serialization utilities for SpacetimeDB storage.

Provides functions for serializing/deserializing PyTorch tensors,
game trajectories, and training checkpoints to/from bytes for
storage in SpacetimeDB blob columns.

Reuses existing serialization patterns from GameTrajectory.to_bytes()
and TrainingExample.to_bytes() in distributed/shared_storage.py.
"""

from __future__ import annotations

import io
import json
import logging
import lzma
from typing import TYPE_CHECKING, Any

import torch

from alpha_zero.spacetimedb.exceptions import SpacetimeDBSerializationError

if TYPE_CHECKING:
    from alpha_zero.distributed.shared_storage import GameTrajectory
    from alpha_zero.storage.models import EvaluationResult, ModelMetadata

logger = logging.getLogger(__name__)


def serialize_state_dict(
    state_dict: dict[str, torch.Tensor],
    compress: bool = True,
) -> bytes:
    """Serialize a PyTorch state dict to bytes.

    Args:
        state_dict: Model state dictionary.
        compress: Whether to apply LZMA compression.

    Returns:
        Serialized bytes.

    Raises:
        SpacetimeDBSerializationError: If serialization fails.
    """
    try:
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        data = buffer.getvalue()
        if compress:
            data = lzma.compress(data)
        return data
    except Exception as e:
        raise SpacetimeDBSerializationError(
            f"Failed to serialize state dict: {e}"
        ) from e


def deserialize_state_dict(
    data: bytes,
    compressed: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Deserialize a state dict from bytes.

    Args:
        data: Serialized state dict bytes.
        compressed: Whether data is LZMA compressed.
        map_location: Device to map tensors to.

    Returns:
        Deserialized state dictionary.

    Raises:
        SpacetimeDBSerializationError: If deserialization fails.
    """
    try:
        if compressed:
            data = lzma.decompress(data)
        buffer = io.BytesIO(data)
        return torch.load(buffer, map_location=map_location, weights_only=True)  # type: ignore[no-any-return]
    except Exception as e:
        raise SpacetimeDBSerializationError(
            f"Failed to deserialize state dict: {e}"
        ) from e


def serialize_trajectory(
    trajectory: GameTrajectory,
    compress: bool = False,
) -> bytes:
    """Serialize a GameTrajectory to bytes.

    Delegates to GameTrajectory.to_bytes() for consistency with
    the existing serialization format.

    Args:
        trajectory: Game trajectory to serialize.
        compress: Whether to apply compression.

    Returns:
        Serialized trajectory bytes.

    Raises:
        SpacetimeDBSerializationError: If serialization fails.
    """
    try:
        return trajectory.to_bytes(compress=compress)
    except Exception as e:
        raise SpacetimeDBSerializationError(
            f"Failed to serialize trajectory: {e}"
        ) from e


def deserialize_trajectory(
    data: bytes,
    compressed: bool = False,
) -> GameTrajectory:
    """Deserialize a GameTrajectory from bytes.

    Args:
        data: Serialized trajectory bytes.
        compressed: Whether data is compressed.

    Returns:
        Deserialized GameTrajectory.

    Raises:
        SpacetimeDBSerializationError: If deserialization fails.
    """
    try:
        from alpha_zero.distributed.shared_storage import GameTrajectory

        return GameTrajectory.from_bytes(data, compressed=compressed)
    except Exception as e:
        raise SpacetimeDBSerializationError(
            f"Failed to deserialize trajectory: {e}"
        ) from e


def serialize_checkpoint(
    checkpoint_data: dict[str, Any],
    compress: bool = True,
) -> bytes:
    """Serialize a full training checkpoint to bytes.

    Args:
        checkpoint_data: Checkpoint dictionary (model, optimizer, scheduler, etc.).
        compress: Whether to apply LZMA compression.

    Returns:
        Serialized checkpoint bytes.

    Raises:
        SpacetimeDBSerializationError: If serialization fails.
    """
    try:
        buffer = io.BytesIO()
        torch.save(checkpoint_data, buffer)
        data = buffer.getvalue()
        if compress:
            data = lzma.compress(data)
        return data
    except Exception as e:
        raise SpacetimeDBSerializationError(
            f"Failed to serialize checkpoint: {e}"
        ) from e


def deserialize_checkpoint(
    data: bytes,
    compressed: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Deserialize a training checkpoint from bytes.

    Args:
        data: Serialized checkpoint bytes.
        compressed: Whether data is LZMA compressed.
        map_location: Device to map tensors to.

    Returns:
        Deserialized checkpoint dictionary.

    Raises:
        SpacetimeDBSerializationError: If deserialization fails.
    """
    try:
        if compressed:
            data = lzma.decompress(data)
        buffer = io.BytesIO(data)
        return torch.load(buffer, map_location=map_location, weights_only=False)  # type: ignore[no-any-return]
    except Exception as e:
        raise SpacetimeDBSerializationError(
            f"Failed to deserialize checkpoint: {e}"
        ) from e


def serialize_metadata(metadata: ModelMetadata | EvaluationResult | Any) -> str:
    """Serialize metadata to JSON string.

    Handles dataclasses by converting to dict first.

    Args:
        metadata: Metadata object to serialize.

    Returns:
        JSON string representation.

    Raises:
        SpacetimeDBSerializationError: If serialization fails.
    """
    try:
        from dataclasses import asdict, is_dataclass

        if is_dataclass(metadata) and not isinstance(metadata, type):
            data = asdict(metadata)
        elif isinstance(metadata, dict):
            data = metadata
        else:
            data = {"value": str(metadata)}
        return json.dumps(data, default=str)
    except Exception as e:
        raise SpacetimeDBSerializationError(
            f"Failed to serialize metadata: {e}"
        ) from e


def deserialize_metadata(data: str) -> dict[str, Any]:
    """Deserialize metadata from JSON string.

    Args:
        data: JSON string to deserialize.

    Returns:
        Dictionary representation of the metadata.

    Raises:
        SpacetimeDBSerializationError: If deserialization fails.
    """
    try:
        return json.loads(data)  # type: ignore[no-any-return]
    except Exception as e:
        raise SpacetimeDBSerializationError(
            f"Failed to deserialize metadata: {e}"
        ) from e


def serialize_config(config: dict[str, Any]) -> str:
    """Serialize training configuration to JSON string.

    Args:
        config: Configuration dictionary.

    Returns:
        JSON string.

    Raises:
        SpacetimeDBSerializationError: If serialization fails.
    """
    try:
        return json.dumps(config, default=str, indent=2)
    except Exception as e:
        raise SpacetimeDBSerializationError(
            f"Failed to serialize config: {e}"
        ) from e


def deserialize_config(data: str) -> dict[str, Any]:
    """Deserialize training configuration from JSON string.

    Args:
        data: JSON string.

    Returns:
        Configuration dictionary.

    Raises:
        SpacetimeDBSerializationError: If deserialization fails.
    """
    try:
        return json.loads(data)  # type: ignore[no-any-return]
    except Exception as e:
        raise SpacetimeDBSerializationError(
            f"Failed to deserialize config: {e}"
        ) from e


def estimate_blob_size_mb(data: bytes) -> float:
    """Estimate the size of a blob in megabytes.

    Args:
        data: Bytes data.

    Returns:
        Size in megabytes.
    """
    return len(data) / (1024 * 1024)


__all__ = [
    "deserialize_checkpoint",
    "deserialize_config",
    "deserialize_metadata",
    "deserialize_state_dict",
    "deserialize_trajectory",
    "estimate_blob_size_mb",
    "serialize_checkpoint",
    "serialize_config",
    "serialize_metadata",
    "serialize_state_dict",
    "serialize_trajectory",
]
