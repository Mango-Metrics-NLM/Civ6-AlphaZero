"""MuZero Smoke Test - Path Isolation & Network Creation Validation."""

from __future__ import annotations

import sys


def test_path_isolation() -> None:
    """Verify GCS paths are isolated from production."""
    from alpha_zero.cloud.gcp.config import GCSPathConfig

    paths = GCSPathConfig(agent_id="muzero-smoke")
    prod = GCSPathConfig()

    model = paths.model_path(1)
    traj = paths.trajectory_path(1, 0)
    ckpt = paths.checkpoint_path(100)

    print(f"Smoke model:      {model}")
    print(f"Smoke trajectory: {traj}")
    print(f"Smoke checkpoint: {ckpt}")
    print(f"Prod model:       {prod.model_path(1)}")

    assert "muzero-smoke" in model, f"Model not isolated: {model}"
    assert "muzero-smoke" in traj, f"Trajectory not isolated: {traj}"
    assert "muzero-smoke" in ckpt, f"Checkpoint not isolated: {ckpt}"
    assert prod.model_path(1) != model, "DANGER: model paths collide!"
    assert prod.trajectory_path(1, 0) != traj, "DANGER: trajectory paths collide!"

    print("\nPath isolation verified.")


def test_muzero_network() -> None:
    """Verify MuZeroNetwork can be instantiated and performs a forward pass."""
    import torch

    from alpha_zero.core.muzero_network import MuZeroConfig, MuZeroNetwork

    config = MuZeroConfig(
        spatial_channels=13,
        spatial_size=6,
        vector_dim=64,
        action_space_size=376,
        num_res_blocks=2,
        hidden_state_channels=32,
    )
    model = MuZeroNetwork(config=config)
    model.eval()

    spatial = torch.randn(1, 13, 6, 6)
    vector = torch.randn(1, 64)
    with torch.no_grad():
        policy, value = model(spatial, vector)

    params = sum(p.numel() for p in model.parameters())
    print(f"\nMuZeroNetwork: {params:,} params")
    print(f"Policy shape: {policy.shape}, Value shape: {value.shape}")
    print("MuZero network smoke test passed.")


if __name__ == "__main__":
    print("=== MuZero Dry-Run Smoke Tests ===\n")

    print("[1/2] GCS Path Isolation...")
    test_path_isolation()

    print("\n[2/2] MuZero Network Creation...")
    test_muzero_network()

    print("\n=== All dry-run smoke tests PASSED ===")
    sys.exit(0)
