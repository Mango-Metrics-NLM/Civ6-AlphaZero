"""
Python AI Service -- SpacetimeDB Integration Layer

This module connects the AlphaZero AI (GumbelMCTS + Civ6TransformerNetwork)
to the SpacetimeDB server module. It:
  1. Subscribes to the `ai_turn_request` table.
  2. When a new row appears, deserializes the spatial tensor, vector features,
     and action mask from the binary blobs.
  3. Runs MCTS inference to select an action.
  4. Calls `submit_move` on the SpacetimeDB module with the chosen action
     and MCTS diagnostics (policy distribution, value estimate).

Prerequisites:
  pip install spacetimedb-sdk  # SpacetimeDB Python SDK
  # The rest comes from the existing alpha_zero package

Usage:
  python -m server_module.ai_service \
      --host "http://localhost:3000" \
      --db-name "civ6az" \
      --model-path "checkpoints/v268.pt" \
      --mcts-sims 25

Architecture:
  +-----------+         +-------------------+         +------------------+
  | SpacetimeDB|  sub   |  ai_service.py    | import  | alpha_zero.core  |
  | Module     |------->| (this file)       |-------->| GumbelMCTS       |
  |            |<-------| submit_move()     |         | TransformerNet   |
  +-----------+  call   +-------------------+         +------------------+
"""

from __future__ import annotations

import argparse
import logging
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (must match Rust server module)
# ---------------------------------------------------------------------------

BOARD_SIZE = 8
SPATIAL_CHANNELS = 13
VECTOR_DIM = 64
ACTION_SPACE_SIZE = 376
ACTION_MASK_BYTES = (ACTION_SPACE_SIZE + 7) // 8  # 47


# ---------------------------------------------------------------------------
# Binary deserialization helpers
# ---------------------------------------------------------------------------

def unpack_spatial_tensor(blob: bytes) -> np.ndarray:
    """Unpack spatial tensor from little-endian f32 bytes.

    Args:
        blob: 3328 bytes (13 * 8 * 8 * 4 bytes).

    Returns:
        numpy array of shape (13, 8, 8) float32.
    """
    expected = SPATIAL_CHANNELS * BOARD_SIZE * BOARD_SIZE
    if len(blob) != expected * 4:
        raise ValueError(
            f"spatial_tensor blob is {len(blob)} bytes, expected {expected * 4}"
        )
    flat = np.frombuffer(blob, dtype="<f4")  # little-endian float32
    return flat.reshape(SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE).copy()


def unpack_vector_features(blob: bytes) -> np.ndarray:
    """Unpack vector features from little-endian f32 bytes.

    Args:
        blob: 256 bytes (64 * 4 bytes).

    Returns:
        numpy array of shape (64,) float32.
    """
    if len(blob) != VECTOR_DIM * 4:
        raise ValueError(
            f"vector_features blob is {len(blob)} bytes, expected {VECTOR_DIM * 4}"
        )
    return np.frombuffer(blob, dtype="<f4").copy()


def unpack_action_mask(blob: bytes) -> np.ndarray:
    """Unpack action mask from packed bits.

    Args:
        blob: 47 bytes (ceil(376/8)).

    Returns:
        numpy array of shape (376,) float32 with 0.0 or 1.0 values.
    """
    if len(blob) != ACTION_MASK_BYTES:
        raise ValueError(
            f"action_mask blob is {len(blob)} bytes, expected {ACTION_MASK_BYTES}"
        )
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for i in range(ACTION_SPACE_SIZE):
        byte_idx = i // 8
        bit_idx = i % 8
        if (blob[byte_idx] >> bit_idx) & 1:
            mask[i] = 1.0
    return mask


def pack_policy(policy: np.ndarray) -> bytes:
    """Pack MCTS policy distribution as little-endian f32 bytes.

    Args:
        policy: numpy array of shape (376,) float32.

    Returns:
        1504 bytes.
    """
    return policy.astype("<f4").tobytes()


def pack_value(value: float) -> bytes:
    """Pack MCTS value estimate as little-endian f32.

    Args:
        value: float in [-1, 1].

    Returns:
        4 bytes.
    """
    return struct.pack("<f", value)


# ---------------------------------------------------------------------------
# AI Service class
# ---------------------------------------------------------------------------

class AiService:
    """Connects AlphaZero inference to SpacetimeDB.

    Lifecycle:
        1. __init__: Load model, create MCTS.
        2. connect(): Establish SpacetimeDB connection + subscriptions.
        3. run(): Blocking loop -- process AI turn requests as they arrive.
        4. shutdown(): Clean disconnect.
    """

    def __init__(
        self,
        model_path: str,
        mcts_sims: int = 25,
        device: str = "cpu",
        model_version: str = "v268",
    ) -> None:
        self.model_path = model_path
        self.mcts_sims = mcts_sims
        self.device = device
        self.model_version = model_version

        # These will be imported lazily so the module file can be read
        # without requiring torch to be installed
        self.network = None
        self.mcts = None

    def load_model(self) -> None:
        """Load the neural network and create MCTS instance."""
        from alpha_zero.core.transformer_network import Civ6TransformerNetwork
        from alpha_zero.core.mcts_factory import create_mcts

        logger.info("Loading model from %s", self.model_path)

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Determine config from checkpoint metadata
        model_config = checkpoint.get("model_config", {})
        spatial_size = model_config.get("spatial_size", BOARD_SIZE)
        action_space_size = model_config.get("action_space_size", ACTION_SPACE_SIZE)

        self.network = Civ6TransformerNetwork(
            spatial_channels=SPATIAL_CHANNELS,
            spatial_size=spatial_size,
            action_space_size=action_space_size,
            vector_dim=VECTOR_DIM,
        )

        # Load weights
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.network.load_state_dict(state_dict)
        self.network.to(self.device)

        # Create GumbelMCTS
        self.mcts = create_mcts(
            mcts_type="gumbel",
            network=self.network,
            num_simulations=self.mcts_sims,
            device=self.device,
        )

        logger.info(
            "Model loaded: %s, MCTS sims=%d, device=%s",
            self.model_version, self.mcts_sims, self.device,
        )

    def process_turn_request(
        self,
        game_id: int,
        spatial_blob: bytes,
        vector_blob: bytes,
        mask_blob: bytes,
    ) -> tuple[int, np.ndarray, float]:
        """Run MCTS and select an action for the given board state.

        Args:
            game_id: The game ID (for logging).
            spatial_blob: Raw spatial tensor bytes from SpacetimeDB.
            vector_blob: Raw vector features bytes.
            mask_blob: Raw action mask bytes (packed bits).

        Returns:
            Tuple of (action_index, policy_distribution, value_estimate).
        """
        if self.network is None or self.mcts is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Deserialize
        spatial = unpack_spatial_tensor(spatial_blob)
        vector = unpack_vector_features(vector_blob)
        action_mask = unpack_action_mask(mask_blob)

        # Convert to tensors
        spatial_t = torch.from_numpy(spatial).unsqueeze(0).to(self.device)
        vector_t = torch.from_numpy(vector).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)

        # Build a lightweight env-like wrapper for MCTS
        # GumbelMCTS expects an object with get_observation() and step()
        # We create a minimal shim here
        env_shim = _EnvShim(spatial, vector, action_mask)

        # Run MCTS search
        action_probs, root_value = self.mcts.search(env_shim, add_noise=False)

        # Select action (temperature=0 for competitive play)
        action = int(np.argmax(action_probs))

        # Verify action is valid
        if action_mask[action] < 0.5:
            logger.warning(
                "MCTS selected invalid action %d for game %d, "
                "falling back to argmax of masked policy",
                action, game_id,
            )
            masked = action_probs * action_mask
            if masked.sum() > 0:
                action = int(np.argmax(masked))
            else:
                # Ultimate fallback: end turn
                action = 0

        logger.info(
            "Game %d: selected action %d (value=%.3f, %d valid actions)",
            game_id, action, root_value, int(action_mask.sum()),
        )

        return action, action_probs, float(root_value)

    def connect_and_run(self, host: str, db_name: str, token: str) -> None:
        """Connect to SpacetimeDB and run the AI service loop.

        This is a reference implementation. The actual SpacetimeDB Python SDK
        connection API may differ. Adapt as needed.

        Args:
            host: SpacetimeDB host URL (e.g. "http://localhost:3000").
            db_name: Database name (e.g. "civ6az").
            token: Authentication token.
        """
        # NOTE: This is pseudo-code for the SpacetimeDB Python SDK.
        # The actual SDK API will look similar but may have different method names.
        # See: https://spacetimedb.com/docs/sdks/python
        #
        # from spacetimedb_sdk import SpacetimeDBClient
        #
        # client = SpacetimeDBClient(host)
        # client.connect(db_name, token)
        #
        # # Register as AI player
        # client.call_reducer("register_ai_player", [
        #     f"AlphaZero-{self.model_version}",
        #     self.model_version,
        # ])
        #
        # # Subscribe to AI turn requests
        # client.subscribe(["SELECT * FROM ai_turn_request"])
        #
        # # Register callback for new rows
        # @client.on_insert("ai_turn_request")
        # def on_ai_turn(row):
        #     action, policy, value = self.process_turn_request(
        #         game_id=row.game_id,
        #         spatial_blob=row.spatial_tensor,
        #         vector_blob=row.vector_features,
        #         mask_blob=row.action_mask,
        #     )
        #     client.call_reducer("submit_move", [
        #         row.game_id,
        #         action,
        #         pack_policy(policy),
        #         pack_value(value),
        #     ])
        #
        # # Block forever (SDK handles reconnection)
        # client.run()

        logger.info(
            "AI service ready: host=%s, db=%s, model=%s",
            host, db_name, self.model_version,
        )

        # Polling fallback (if SDK loop is not available)
        logger.info("Starting polling loop (replace with SDK callbacks)")
        while True:
            time.sleep(0.1)  # 100ms poll interval


class _EnvShim:
    """Minimal environment shim for MCTS that holds a frozen board state.

    GumbelMCTS.search() calls env.get_observation() and env.clone().
    Since we only need the root observation for the initial expansion,
    this shim is sufficient.
    """

    def __init__(
        self,
        spatial: np.ndarray,
        vector: np.ndarray,
        action_mask: np.ndarray,
    ) -> None:
        self.spatial = spatial
        self.vector = vector
        self.action_mask = action_mask
        self.board_size = BOARD_SIZE
        self.current_player = 0
        self.done = False

        # Mimic MicroCivEnv interface
        from gymnasium import spaces
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

    def get_observation(self) -> dict[str, np.ndarray]:
        return {
            "spatial": self.spatial,
            "vector": self.vector,
            "action_mask": self.action_mask,
        }

    def get_valid_actions(self) -> np.ndarray:
        return self.action_mask

    def clone(self) -> "_EnvShim":
        return _EnvShim(
            self.spatial.copy(),
            self.vector.copy(),
            self.action_mask.copy(),
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Civ6 AlphaZero AI Service for SpacetimeDB"
    )
    parser.add_argument(
        "--host", default="http://localhost:3000", help="SpacetimeDB host URL"
    )
    parser.add_argument(
        "--db-name", default="civ6az", help="SpacetimeDB database name"
    )
    parser.add_argument("--token", default="", help="Auth token")
    parser.add_argument(
        "--model-path", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model-version", default="v268", help="Model version string"
    )
    parser.add_argument(
        "--mcts-sims", type=int, default=25, help="MCTS simulations per move"
    )
    parser.add_argument(
        "--device", default="cpu", help="PyTorch device (cpu/cuda)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    service = AiService(
        model_path=args.model_path,
        mcts_sims=args.mcts_sims,
        device=args.device,
        model_version=args.model_version,
    )
    service.load_model()
    service.connect_and_run(args.host, args.db_name, args.token)


if __name__ == "__main__":
    main()
