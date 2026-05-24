"""Build a synthetic-weights champion checkpoint for sprint smoke tests.

W1.4 of the v1.7.0 sprint plan: this is the **primary fallback path** for
integration tests and the W3.2 demo when a real trained champion is not
available. It instantiates a ``MicroCivNetwork`` with default architecture
parameters, saves random-init weights via ``LocalStorageBackend``, and
returns the path so callers can wire ``BridgeAgentLoop`` against a real
on-disk artifact without waiting on a training run.

The weights are NOT useful for play quality - they exercise the wiring,
the file IPC, and the MLflow harness only. Real training is W3.2's job.

Public API:
    build_smoke_champion(storage_dir, *, board_size=8, num_players=2,
                          version=0, iteration=0) -> Path

Used by:
    - tests/integration/test_civ6_bridge_e2e.py (smoke-test fixture)
    - tests/unit/test_bridge_agent_loop.py (when load_champion is needed)
    - W3.2 manual demo as a fallback when training has not yet produced a
      champion at ``checkpoints/v1.7.0-smoke/champion/``
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.storage.config import StorageConfig
from alpha_zero.storage.factory import create_storage_backend
from alpha_zero.storage.models import ModelMetadata

logger = logging.getLogger(__name__)

# Smoke-test defaults. Kept aligned with civ6_local.py / MicroCivEnv defaults so
# the fixture is binary-compatible with the agent-loop wiring.
DEFAULT_BOARD_SIZE = 8
DEFAULT_NUM_PLAYERS = 2
DEFAULT_VERSION = 0
DEFAULT_ITERATION = 0


def build_smoke_champion(
    storage_dir: Path,
    *,
    board_size: int = DEFAULT_BOARD_SIZE,
    num_players: int = DEFAULT_NUM_PLAYERS,
    version: int = DEFAULT_VERSION,
    iteration: int = DEFAULT_ITERATION,
) -> Path:
    """Build a synthetic-weights champion in ``storage_dir`` and return its path.

    Idempotent: overwrites any existing champion in ``storage_dir``. Callers
    that need a fresh-only fixture should pass a ``tmp_path``.

    Args:
        storage_dir: Base storage directory (will contain
            ``models/champion/model.pt`` + ``metadata.json``).
        board_size: Board size the network is dimensioned for. Must match the
            ``board_size`` used by the consumer's env.
        num_players: 2 keeps the scalar value head; >2 swaps to the N-player
            value head. Tests should use 2 unless specifically exercising the
            N-player path.
        version: Model version recorded in metadata.
        iteration: Training iteration recorded in metadata.

    Returns:
        Absolute path to the saved ``model.pt`` file.

    Raises:
        ValueError: If ``board_size`` or ``num_players`` is out of range.
    """
    if board_size < 4 or board_size > 64:
        raise ValueError(f"board_size must be 4-64, got {board_size}")
    if num_players < 2 or num_players > 8:
        raise ValueError(f"num_players must be 2-8, got {num_players}")

    storage_dir = Path(storage_dir).expanduser().resolve()
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Spatial channel count must match MicroCivEncoder convention:
    # 13 for 2-player (3 terrain + 3 own-units + 3 enemy-units + own-cities +
    # enemy-cities + visibility + bias), 3 + 4*N + 2 for N-player.
    spatial_channels = 13 if num_players == 2 else 3 + 4 * num_players + 2

    # Action space size: matches MicroCivEnv default (376 actions on 16x16,
    # scales with board size). We use the network's default to stay agnostic.
    network = MicroCivNetwork(
        board_size=board_size,
        spatial_channels=spatial_channels,
        num_players=num_players,
    )

    config = StorageConfig.for_local(str(storage_dir))
    storage = create_storage_backend(config)

    metadata = ModelMetadata(
        iteration=iteration,
        version=version,
        metrics={
            "fixture": "synthetic_weights",
            "board_size": board_size,
            "num_players": num_players,
            "spatial_channels": spatial_channels,
            "fixture_source": "tests/fixtures/build_smoke_champion.py",
        },
    )

    state_dict: dict[str, torch.Tensor] = network.state_dict()
    saved_path_str = storage.save_champion(state_dict, metadata)
    saved_path = Path(saved_path_str)

    logger.info(
        "Built synthetic-weights champion: path=%s board_size=%d num_players=%d "
        "version=%d iteration=%d",
        saved_path,
        board_size,
        num_players,
        version,
        iteration,
    )
    return saved_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a synthetic-weights champion for sprint smoke tests.",
    )
    parser.add_argument(
        "storage_dir",
        type=Path,
        help="Storage base dir; champion saved to <dir>/models/champion/",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=DEFAULT_BOARD_SIZE,
        help=f"Board size (default {DEFAULT_BOARD_SIZE})",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=DEFAULT_NUM_PLAYERS,
        help=f"Number of players (default {DEFAULT_NUM_PLAYERS})",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=DEFAULT_VERSION,
        help=f"Model version recorded in metadata (default {DEFAULT_VERSION})",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=DEFAULT_ITERATION,
        help=f"Iteration recorded in metadata (default {DEFAULT_ITERATION})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Log verbosity (default INFO)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        path = build_smoke_champion(
            args.storage_dir,
            board_size=args.board_size,
            num_players=args.num_players,
            version=args.version,
            iteration=args.iteration,
        )
    except (ValueError, OSError) as exc:
        logger.error("Failed to build smoke champion: %s", exc)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
