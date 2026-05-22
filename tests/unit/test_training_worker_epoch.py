"""Contract test for alpha_zero.cloud.gcp.training_worker.train_epoch.

Pins the existing production train_epoch (training_worker.py:620) so a future
refactor cannot silently change its convergence behaviour. Calls the function
through its real public signature with optional args set to None.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from alpha_zero.cloud.gcp.training_worker import create_model, train_epoch


def test_train_epoch_reduces_loss_on_overfit_batch() -> None:
    """train_epoch must reduce total loss when called repeatedly on the same batch."""
    model = create_model(
        board_size=8,
        action_space_size=376,
        num_res_blocks=2,
        num_filters=64,
    )
    device = torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train_epoch internally reshapes spatial from flat (B, 13*8*8) to (B, 13, 8, 8).
    spatial = torch.randn(8, 13 * 8 * 8)
    vector = torch.randn(8, 64)
    action_mask = torch.ones(8, 376)
    policies = torch.softmax(torch.randn(8, 376), dim=-1)
    values = torch.randn(8)
    loader = DataLoader(
        TensorDataset(spatial, vector, action_mask, policies, values),
        batch_size=4,
    )

    total_first, _p_first, _v_first, _n = train_epoch(
        model, loader, optimizer, device,
    )
    total_last = total_first
    for _ in range(10):
        total_last, _p, _v, _n = train_epoch(model, loader, optimizer, device)

    assert total_last < total_first, (
        f"train_epoch did not reduce loss: first={total_first:.4f}, "
        f"last={total_last:.4f}"
    )
