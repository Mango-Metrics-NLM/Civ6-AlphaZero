"""
Unit Tests for Phase 15: Separate Policy/Value Head Learning Rates.

Tests:
- Param group creation with correct names and LRs
- Default multipliers produce uniform LRs
- LR scheduler preserves per-group ratios
- train_epoch works with separate LRs
- CLI arg parsing for multiplier flags
- Orchestrator passes multiplier args only when non-default
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from alpha_zero.cloud.gcp.training_worker import parse_args, train_epoch
from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig


@pytest.fixture
def model_and_config():
    """Create a minimal transformer model for testing param groups."""
    config = TransformerConfig(
        spatial_channels=13,
        spatial_size=8,
        vector_dim=64,
        action_space_size=376,
        embed_dim=64,
        num_layers=2,
        num_heads=8,
    )
    model = Civ6TransformerNetwork(config=config)
    return model, config


@pytest.fixture
def training_setup(model_and_config):
    """Create model, optimizer with param groups, and dataloader."""
    model, _config = model_and_config

    batch_size = 2
    num_examples = 4
    spatial = torch.randn(num_examples, 13 * 8 * 8)
    vector = torch.randn(num_examples, 64)
    policies = torch.softmax(torch.randn(num_examples, 376), dim=-1)
    values = torch.randn(num_examples)

    dataset = TensorDataset(spatial, vector, policies, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cpu")

    return model, dataloader, device


class TestParamGroupCreation:
    """Test optimizer parameter group construction."""

    def test_param_groups_created(self, model_and_config):
        """3 param groups with correct names and LRs when value_lr_multiplier=0.1."""
        model, _ = model_and_config
        base_lr = 0.001
        value_mult = 0.1
        policy_mult = 1.0

        param_groups = [
            {
                "params": list(model.encoder.parameters()),
                "lr": base_lr,
                "base_lr_ratio": 1.0,
                "name": "encoder",
            },
            {
                "params": list(model.policy_head.parameters()),
                "lr": base_lr * policy_mult,
                "base_lr_ratio": policy_mult,
                "name": "policy_head",
            },
            {
                "params": list(model.value_head.parameters()),
                "lr": base_lr * value_mult,
                "base_lr_ratio": value_mult,
                "name": "value_head",
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        assert len(optimizer.param_groups) == 3
        assert optimizer.param_groups[0]["name"] == "encoder"
        assert optimizer.param_groups[1]["name"] == "policy_head"
        assert optimizer.param_groups[2]["name"] == "value_head"

        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
        assert optimizer.param_groups[1]["lr"] == pytest.approx(0.001)
        assert optimizer.param_groups[2]["lr"] == pytest.approx(0.0001)

        # All model params should be covered
        total_group_params = sum(
            sum(p.numel() for p in pg["params"]) for pg in optimizer.param_groups
        )
        total_model_params = sum(p.numel() for p in model.parameters())
        assert total_group_params == total_model_params

    def test_default_multipliers_equivalent(self, model_and_config):
        """Both multipliers=1.0 produces same LR for all groups."""
        model, _ = model_and_config
        base_lr = 0.001

        param_groups = [
            {"params": list(model.encoder.parameters()), "lr": base_lr, "base_lr_ratio": 1.0},
            {"params": list(model.policy_head.parameters()), "lr": base_lr, "base_lr_ratio": 1.0},
            {"params": list(model.value_head.parameters()), "lr": base_lr, "base_lr_ratio": 1.0},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        for pg in optimizer.param_groups:
            assert pg["lr"] == pytest.approx(base_lr)


class TestSchedulerPreservesRatios:
    """Test that LR scheduler updates preserve per-group LR ratios."""

    def test_scheduler_preserves_ratios(self, model_and_config):
        """After LR scheduler step, value head LR stays 0.1x encoder LR."""
        model, _ = model_and_config
        base_lr = 0.001
        value_mult = 0.1

        param_groups = [
            {"params": list(model.encoder.parameters()), "lr": base_lr, "base_lr_ratio": 1.0},
            {"params": list(model.policy_head.parameters()), "lr": base_lr, "base_lr_ratio": 1.0},
            {
                "params": list(model.value_head.parameters()),
                "lr": base_lr * value_mult,
                "base_lr_ratio": value_mult,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        # Simulate what train_epoch does when applying scheduled LR
        scheduled_lr = 0.0005  # Simulating a cosine decay step
        for param_group in optimizer.param_groups:
            ratio = param_group.get("base_lr_ratio", 1.0)
            param_group["lr"] = scheduled_lr * ratio

        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0005)
        assert optimizer.param_groups[1]["lr"] == pytest.approx(0.0005)
        assert optimizer.param_groups[2]["lr"] == pytest.approx(0.00005)

        # Ratio is preserved
        assert optimizer.param_groups[2]["lr"] / optimizer.param_groups[0]["lr"] == pytest.approx(
            value_mult
        )


class TestTrainEpochSeparateLRs:
    """Test train_epoch with separate LRs completes without error."""

    def test_train_epoch_separate_lrs(self, training_setup):
        """Mini training loop completes with separate per-head LRs."""
        model, dataloader, device = training_setup
        base_lr = 0.001
        value_mult = 0.1

        param_groups = [
            {
                "params": list(model.encoder.parameters()),
                "lr": base_lr,
                "base_lr_ratio": 1.0,
                "name": "encoder",
            },
            {
                "params": list(model.policy_head.parameters()),
                "lr": base_lr,
                "base_lr_ratio": 1.0,
                "name": "policy_head",
            },
            {
                "params": list(model.value_head.parameters()),
                "lr": base_lr * value_mult,
                "base_lr_ratio": value_mult,
                "name": "value_head",
            },
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        total_loss, policy_loss, value_loss, num_batches = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            global_step=0,
            wandb_logger=None,
            spatial_channels=13,
            spatial_size=8,
        )

        assert num_batches == 2
        assert total_loss > 0
        assert policy_loss > 0
        assert value_loss > 0

        # Verify LRs unchanged (no scheduler)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr)
        assert optimizer.param_groups[2]["lr"] == pytest.approx(base_lr * value_mult)


class TestParseArgsMultipliers:
    """Test CLI arg parsing for multiplier flags."""

    def test_parse_args_multiplier_flags(self):
        """CLI args parse correctly with explicit values."""
        test_args = [
            "training_worker.py",
            "--iteration=1",
            "--trajectory_iterations=1",
            "--value_lr_multiplier=0.1",
            "--policy_lr_multiplier=0.5",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.value_lr_multiplier == 0.1
        assert args.policy_lr_multiplier == 0.5

    def test_parse_args_multiplier_defaults(self):
        """Multiplier defaults are 1.0."""
        test_args = [
            "training_worker.py",
            "--iteration=1",
            "--trajectory_iterations=1",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.value_lr_multiplier == 1.0
        assert args.policy_lr_multiplier == 1.0


class TestOrchestratorPassThrough:
    """Test orchestrator passes multiplier args only when non-default."""

    def test_orchestrator_passes_multiplier_args(self):
        """build_training_extra_args includes multiplier args only when non-default."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import build_training_extra_args

        # Create mock args with non-default multipliers
        args = MagicMock()
        args.board_size = 8
        args.action_space_size = 376
        args.num_res_blocks = 6
        args.num_filters = 64
        args.lr_schedule_type = "constant"
        args.weight_decay = 0.01
        args.value_lr_multiplier = 0.1
        args.policy_lr_multiplier = 1.0

        extra = build_training_extra_args(args)

        assert "--value_lr_multiplier=0.1" in extra
        assert not any("policy_lr_multiplier" in a for a in extra)

    def test_orchestrator_skips_default_multipliers(self):
        """build_training_extra_args omits multipliers when both are 1.0."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import build_training_extra_args

        args = MagicMock()
        args.board_size = 8
        args.action_space_size = 376
        args.num_res_blocks = 6
        args.num_filters = 64
        args.lr_schedule_type = "constant"
        args.weight_decay = 0.01
        args.value_lr_multiplier = 1.0
        args.policy_lr_multiplier = 1.0

        extra = build_training_extra_args(args)

        assert not any("value_lr_multiplier" in a for a in extra)
        assert not any("policy_lr_multiplier" in a for a in extra)
