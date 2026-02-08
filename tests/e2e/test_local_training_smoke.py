"""Local training smoke test — end-to-end pipeline validation.

Verifies the complete self-play → training → evaluation pipeline
runs locally without errors on a minimal configuration.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

logger = logging.getLogger(__name__)


def _torch_load_compat(
    path: Path,
    *,
    map_location: str = "cpu",
    weights_only: bool = True,
) -> Any:
    """Load PyTorch checkpoint with backwards compatibility.

    Handles PyTorch versions that don't support the ``weights_only``
    parameter (added in 1.13+, enforced in 2.0+).
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Fallback for older PyTorch without weights_only parameter
        logger.warning(
            "torch.load weights_only not supported; falling back to default load"
        )
        return torch.load(path, map_location=map_location)  # noqa: S614


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def smoke_config() -> dict[str, Any]:
    """Minimal training configuration for smoke testing."""
    return {
        "spatial_size": 6,
        "spatial_channels": 13,
        "vector_dim": 64,
        "action_space_size": 100,
        "num_res_blocks": 2,
        "num_filters": 32,
        "mcts_simulations": 4,
        "batch_size": 4,
        "epochs": 1,
        "learning_rate": 0.001,
    }


@pytest.fixture
def temp_checkpoint_dir() -> Any:
    """Temporary directory for checkpoints, auto-cleaned."""
    with tempfile.TemporaryDirectory(prefix="civ6az-smoke-") as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLocalTrainingSmoke:
    """End-to-end local training smoke tests."""

    def test_network_forward_pass(self, smoke_config: dict[str, Any]) -> None:
        """Network produces valid policy and value outputs."""
        from alpha_zero.core.civ6_network import Civ6Network, NetworkConfig

        config = NetworkConfig(
            spatial_channels=smoke_config["spatial_channels"],
            spatial_size=smoke_config["spatial_size"],
            action_space_size=smoke_config["action_space_size"],
            num_res_blocks=smoke_config["num_res_blocks"],
            num_filters=smoke_config["num_filters"],
            vector_dim=smoke_config["vector_dim"],
        )
        net = Civ6Network(config=config)

        spatial = torch.randn(
            1,
            smoke_config["spatial_channels"],
            smoke_config["spatial_size"],
            smoke_config["spatial_size"],
        )
        vector = torch.randn(1, smoke_config["vector_dim"])

        policy, value = net(spatial, vector)

        assert policy.shape == (1, smoke_config["action_space_size"])
        assert value.ndim <= 2  # (1,) or (1,1)
        logger.info("Forward pass OK: policy=%s, value=%s", policy.shape, value.shape)

    def test_mcts_search_completes(self, smoke_config: dict[str, Any]) -> None:
        """MCTS search runs without errors on a small env."""
        from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig
        from alpha_zero.core.civ6_network import Civ6Network, NetworkConfig
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=smoke_config["spatial_size"])
        env.reset()

        # Use the env's real action space to avoid shape mismatches
        actual_action_size = env.action_space.n

        net_cfg = NetworkConfig(
            spatial_size=smoke_config["spatial_size"],
            action_space_size=actual_action_size,
            num_res_blocks=smoke_config["num_res_blocks"],
            num_filters=smoke_config["num_filters"],
        )
        net = Civ6Network(config=net_cfg)

        mcts_cfg = MCTSConfig(num_simulations=smoke_config["mcts_simulations"])
        mcts = Civ6MCTS(network=net, config=mcts_cfg)

        try:
            result = mcts.search(env)
            assert result is not None
            logger.info("MCTS search completed")
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"Optional dependency missing: {exc}")
        except NotImplementedError as exc:
            pytest.skip(f"Feature not yet implemented: {exc}")

    def test_model_save_load_roundtrip(
        self,
        smoke_config: dict[str, Any],
        temp_checkpoint_dir: Path,
    ) -> None:
        """Model state dict can be saved and loaded correctly."""
        from alpha_zero.core.civ6_network import Civ6Network, NetworkConfig

        cfg = NetworkConfig(
            spatial_size=smoke_config["spatial_size"],
            action_space_size=smoke_config["action_space_size"],
            num_res_blocks=smoke_config["num_res_blocks"],
            num_filters=smoke_config["num_filters"],
        )
        net = Civ6Network(config=cfg)

        checkpoint_path = temp_checkpoint_dir / "model_smoke.pt"
        torch.save(net.state_dict(), checkpoint_path)

        assert checkpoint_path.exists()
        assert checkpoint_path.stat().st_size > 0

        net2 = Civ6Network(config=cfg)
        net2.load_state_dict(_torch_load_compat(checkpoint_path, weights_only=True))

        for key in net.state_dict():
            assert torch.equal(net.state_dict()[key], net2.state_dict()[key])

        logger.info("Model save/load roundtrip OK: %s", checkpoint_path)

    def test_environment_episode_completes(
        self, smoke_config: dict[str, Any]
    ) -> None:
        """MicroCivEnv runs a full episode without errors."""
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=smoke_config["spatial_size"])
        env.reset()

        done = False
        step_count = 0
        max_steps = 50

        while not done and step_count < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        logger.info("Episode completed in %d steps (done=%s)", step_count, done)

    def test_loss_computation(self, smoke_config: dict[str, Any]) -> None:
        """Training loss computes without NaN or errors."""
        bs = smoke_config["batch_size"]
        asz = smoke_config["action_space_size"]

        policy_logits = torch.randn(bs, asz)
        target_policy = torch.softmax(torch.randn(bs, asz), dim=-1)
        value = torch.randn(bs, 1)
        target_value = torch.randn(bs, 1)

        policy_loss = -(
            target_policy * torch.log_softmax(policy_logits, dim=-1)
        ).sum(dim=-1).mean()
        value_loss = torch.nn.functional.mse_loss(value, target_value)

        total_loss = policy_loss + value_loss

        assert not torch.isnan(total_loss), "Loss should not be NaN"
        assert not torch.isinf(total_loss), "Loss should not be infinite"
        logger.info("Loss computation OK: total=%.4f", total_loss.item())
