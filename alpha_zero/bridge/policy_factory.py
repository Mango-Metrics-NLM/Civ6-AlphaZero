"""Builders for the policies the real-game bridge runs (GM-B6b).

The two builders ``civ6_local`` used to hold, and :func:`build_search_policy`,
which builds the search with a network of the *simulator's* shape from a
training checkpoint. The search plays in MicroCiv, so its network takes the
simulator's observation (13 channels) at the board and horizon the checkpoint
was trained with; a checkpoint that does not record its horizon is refused.

``civ6_local --mode civ6 --mcts`` is still refused and does not call
:func:`build_search_policy`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_zero.utils.constants import DEFAULT_ACTION_SPACE_SIZE
from alpha_zero.utils.device import resolve_device, resolve_device_str

logger = logging.getLogger(__name__)

_MAX_SAMPLED_ACTIONS = 16


def build_bridge_policy(config: Any, network: Any, *, log: logging.Logger = logger) -> Any:
    """The policy the bridge agent loop runs: the search, or the greedy network.

    Args:
        config: ``Civ6LocalConfig``; ``use_mcts`` selects the search.
        network: The network :func:`build_network_for_bridge` built.
        log: Logger to report through.

    Returns:
        A ``PolicyEngine`` for ``BridgeAgentLoop``.
    """
    if config.use_mcts:
        from alpha_zero.bridge.mcts_policy import Civ6MctsPolicy
        from alpha_zero.core.civ6_mcts import GumbelMCTSConfig

        gumbel_cfg = GumbelMCTSConfig(
            num_simulations=config.num_simulations,
            num_sampled_actions=min(config.num_simulations, 16),
        )
        log.info("Bridge using Civ6MctsPolicy (sims=%d)", config.num_simulations)
        return Civ6MctsPolicy(network=network, config=gumbel_cfg, device=config.device)

    from alpha_zero.bridge.agent_loop import GreedyNetworkPolicy

    log.info("Bridge using GreedyNetworkPolicy")
    return GreedyNetworkPolicy(network=network, device=config.device)


def build_network_for_bridge(
    config: Any,
    action_space_size: int = DEFAULT_ACTION_SPACE_SIZE,
    *,
    log: logging.Logger = logger,
) -> Any:
    """Construct (and optionally load) the full-game policy network for the bridge.

    Returns a ``MicroCivNetwork`` with the full encoder's input shape.
    """
    from alpha_zero.core.checkpoint_utils import safe_torch_load
    from alpha_zero.core.civ6_network import MicroCivNetwork
    from alpha_zero.envs.civ6_encoder import EncoderConfig

    device = resolve_device(config.device)
    # MicroCivNetwork picks the right value head (scalar vs N-player) based
    # on num_players; the bridge currently targets 2-player real Civ6.
    #
    # StateConverter (used by BridgeAgentLoop) encodes states through the
    # full Civ6StateEncoder, which emits EncoderConfig's spatial_channels=84
    # / vector_dim=512 tensors - NOT MicroCivNetwork's own 13-channel/64-dim
    # "MicroCiv default". Passing these explicitly here (rather than
    # changing MicroCivNetwork's defaults, which are still used elsewhere
    # e.g. the microciv/web training modes) keeps the bridge network's
    # input dims matched to what StateConverter.convert_and_encode actually
    # produces.
    encoder_defaults = EncoderConfig()
    network = MicroCivNetwork(
        board_size=config.board_size,
        num_players=config.num_players,
        action_space_size=action_space_size,
        spatial_channels=encoder_defaults.spatial_channels,
        vector_dim=encoder_defaults.vector_dim,
    ).to(device)

    if config.checkpoint_path:
        try:
            from alpha_zero.storage.config import StorageConfig
            from alpha_zero.storage.factory import create_storage_backend

            ckpt_path = Path(config.checkpoint_path)
            # Two supported layouts:
            # 1. Direct .pt file - safe torch.load (weights_only=True; legacy
            #    pickles need CIV6AZ_ALLOW_UNSAFE_CHECKPOINT=1)
            # 2. Storage base dir containing models/champion/model.pt - use
            #    LocalStorageBackend so metadata + future GCS storage work.
            if ckpt_path.is_file():
                state_dict = safe_torch_load(ckpt_path, map_location=device)
                if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                network.load_state_dict(state_dict)
                log.info("Loaded checkpoint from file: %s", ckpt_path)
            elif ckpt_path.is_dir():
                storage = create_storage_backend(StorageConfig.for_local(str(ckpt_path)))
                state_dict, metadata = storage.load_champion(map_location=device)
                network.load_state_dict(state_dict)
                log.info(
                    "Loaded champion v%d from storage: %s",
                    metadata.version,
                    ckpt_path,
                )
            else:
                log.warning(
                    "Checkpoint path %s does not exist - using synthetic weights",
                    ckpt_path,
                )
        except Exception as exc:
            log.warning(
                "Failed to load checkpoint %s: %s - falling back to synthetic weights",
                config.checkpoint_path,
                exc,
            )
    else:
        log.warning(
            "No --checkpoint supplied; bridge will run with synthetic weights "
            "(agent will play poorly). Pass --checkpoint <path> to load a real "
            "trained model."
        )

    return network


@dataclass(frozen=True)
class SearchPolicy:
    """A search policy and the simulator shape it was built for.

    Attributes:
        policy: The ``Civ6MctsPolicy``.
        board_size: The board its network takes; the state converter must use it.
        max_turns: The horizon the simulator plays to.
        trained: Whether the weights came from a checkpoint.
    """

    policy: Any
    board_size: int
    max_turns: int
    trained: bool


def _checkpoint_network(path: Path, device: str) -> tuple[Any, int, int]:
    """(network, board size, horizon) of a training checkpoint."""
    from alpha_zero.core.checkpoint_utils import safe_torch_load
    from alpha_zero.utils.checkpoint import (
        build_micro_network_from_checkpoint,
        extract_state_dict,
    )

    if not path.is_file():
        raise ValueError(f"checkpoint {path} is not a file")
    checkpoint = safe_torch_load(path, map_location=device)
    metadata = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
    horizon = metadata.get("max_turns") if isinstance(metadata, dict) else None
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError(
            f"checkpoint {path} does not record max_turns (its horizon); the search "
            "cannot set the simulator's clock for it"
        )
    network = build_micro_network_from_checkpoint(
        checkpoint,
        default_board_size=0,
        default_action_size=None,
        expected_max_turns=horizon,
        source=path,
    )
    state = extract_state_dict(checkpoint) if isinstance(checkpoint, dict) else checkpoint
    network.load_state_dict(state)
    board = metadata.get("board_size") if isinstance(metadata, dict) else None
    if not isinstance(board, int) or isinstance(board, bool) or board <= 0:
        board = int(network.config.spatial_size)
    return network, board, horizon


def build_search_policy(
    checkpoint_path: str | Path | None = None,
    *,
    device: str = "cpu",
    num_simulations: int = 16,
    seed: int | None = None,
    allow_untrained: bool = False,
    board_size: int | None = None,
    max_turns: int | None = None,
) -> SearchPolicy:
    """The search over the simulator, with a network of the simulator's shape.

    Args:
        checkpoint_path: A training checkpoint (``Civ6Network.save``). Its
            recorded board and horizon decide the network and the clock.
        device: Torch device.
        num_simulations: Search budget per decision.
        seed: Base seed of the search; each decision derives its own from it.
        allow_untrained: Without a checkpoint, build a freshly initialised
            network. Has to be asked for in words: such a search proves wiring,
            not play.
        board_size: Board for the untrained network.
        max_turns: Horizon for the untrained network.

    Raises:
        ValueError: No usable checkpoint and no explicit untrained request, or
            a checkpoint that does not record its horizon.
    """
    from alpha_zero.bridge.mcts_policy import Civ6MctsPolicy
    from alpha_zero.core.civ6_mcts import GumbelMCTSConfig

    resolved = resolve_device_str(device)
    if checkpoint_path is not None:
        network, board, horizon = _checkpoint_network(Path(checkpoint_path), resolved)
        trained = True
    else:
        if not allow_untrained:
            raise ValueError(
                "the search needs a checkpoint; an untrained network must be asked for "
                "with allow_untrained=True"
            )
        if board_size is None or max_turns is None:
            raise ValueError("an untrained search network needs board_size and max_turns")
        from alpha_zero.core.civ6_network import MicroCivNetwork
        from alpha_zero.envs.microciv import MicroCivEnv

        actions = MicroCivEnv(board_size=board_size, max_turns=max_turns).action_space.n
        network = MicroCivNetwork(board_size=board_size, action_space_size=int(actions))
        board, horizon, trained = int(board_size), int(max_turns), False
    network.to(resolved).eval()
    config = GumbelMCTSConfig(
        num_simulations=num_simulations,
        num_sampled_actions=min(num_simulations, _MAX_SAMPLED_ACTIONS),
        seed=seed,
    )
    policy = Civ6MctsPolicy(network=network, config=config, device=resolved, max_turns=horizon)
    return SearchPolicy(policy=policy, board_size=board, max_turns=horizon, trained=trained)
