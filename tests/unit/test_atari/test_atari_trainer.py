"""Unit tests for Atari training components."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import torch

from demos.atari.config import AtariTrainingConfig
from demos.atari.trainer import (
    ReplayBuffer,
    ReplayDataset,
    TrainingExample,
    _discount_returns,
    compute_atari_loss,
)


# ============================================================================
# TrainingExample
# ============================================================================


def _make_example(value: float = 0.5) -> TrainingExample:
    return TrainingExample(
        spatial=np.random.rand(4, 84, 84).astype(np.float32),
        vector=np.zeros(32, dtype=np.float32),
        action_mask=np.ones(6, dtype=np.float32),
        policy=np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1], dtype=np.float32),
        value=value,
    )


# ============================================================================
# ReplayBuffer
# ============================================================================


class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(max_size=100, min_size=5)
        assert len(buf) == 0
        buf.add(_make_example())
        assert len(buf) == 1

    def test_max_size(self):
        buf = ReplayBuffer(max_size=10, min_size=1)
        for _ in range(20):
            buf.add(_make_example())
        assert len(buf) == 10

    def test_sample(self):
        buf = ReplayBuffer(max_size=100, min_size=1)
        for _ in range(50):
            buf.add(_make_example())
        samples = buf.sample(10)
        assert len(samples) == 10
        assert isinstance(samples[0], TrainingExample)

    def test_sample_clamps_to_buffer_size(self):
        buf = ReplayBuffer(max_size=100, min_size=1)
        for _ in range(5):
            buf.add(_make_example())
        samples = buf.sample(100)  # request more than available
        assert len(samples) == 5

    def test_is_ready(self):
        buf = ReplayBuffer(max_size=100, min_size=10)
        assert not buf.is_ready()
        for _ in range(10):
            buf.add(_make_example())
        assert buf.is_ready()

    def test_add_episode(self):
        buf = ReplayBuffer(max_size=100, min_size=1)
        examples = [_make_example() for _ in range(5)]
        buf.add_episode(examples)
        assert len(buf) == 5


# ============================================================================
# ReplayDataset
# ============================================================================


class TestReplayDataset:
    def test_len(self):
        examples = [_make_example() for _ in range(10)]
        ds = ReplayDataset(examples)
        assert len(ds) == 10

    def test_getitem(self):
        examples = [_make_example(0.7)]
        ds = ReplayDataset(examples)
        item = ds[0]
        assert "spatial" in item
        assert "vector" in item
        assert "action_mask" in item
        assert "policy" in item
        assert "value" in item
        assert item["spatial"].shape == (4, 84, 84)
        assert item["value"].item() == pytest.approx(0.7)


# ============================================================================
# Loss function
# ============================================================================


class TestComputeAtariLoss:
    def test_basic_loss(self):
        logits = torch.randn(4, 6)
        value = torch.randn(4)
        target_policy = torch.softmax(torch.randn(4, 6), dim=-1)
        target_value = torch.randn(4)
        total, p_loss, v_loss = compute_atari_loss(logits, value, target_policy, target_value)
        assert total.item() > 0
        assert p_loss.item() > 0
        assert v_loss.item() >= 0

    def test_loss_with_mask(self):
        logits = torch.randn(2, 6)
        value = torch.randn(2)
        target_policy = torch.softmax(torch.randn(2, 6), dim=-1)
        target_value = torch.randn(2)
        mask = torch.tensor([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]], dtype=torch.float32)
        total, p_loss, v_loss = compute_atari_loss(
            logits, value, target_policy, target_value, action_mask=mask
        )
        assert total.item() > 0

    def test_perfect_prediction(self):
        target_policy = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        logits = torch.tensor([[-100.0, 100.0, -100.0, -100.0, -100.0, -100.0]])
        target_value = torch.tensor([0.5])
        value = torch.tensor([0.5])
        total, p_loss, v_loss = compute_atari_loss(logits, value, target_policy, target_value)
        assert p_loss.item() < 0.01
        assert v_loss.item() < 0.01

    def test_gradient_flow(self):
        logits = torch.randn(2, 6, requires_grad=True)
        value = torch.randn(2, requires_grad=True)
        target_policy = torch.softmax(torch.randn(2, 6), dim=-1)
        target_value = torch.randn(2)
        total, _, _ = compute_atari_loss(logits, value, target_policy, target_value)
        total.backward()
        assert logits.grad is not None
        assert value.grad is not None


# ============================================================================
# Discount returns
# ============================================================================


class TestDiscountReturns:
    def test_single_step(self):
        returns = _discount_returns([1.0], gamma=0.99)
        assert len(returns) == 1
        assert returns[0] == pytest.approx(1.0, abs=0.01)

    def test_multi_step(self):
        rewards = [1.0, 1.0, 1.0]
        returns = _discount_returns(rewards, gamma=0.99)
        assert len(returns) == 3
        # Last step should have smallest return
        assert returns[0] > returns[2] or np.isclose(returns[0], returns[2], atol=0.1)

    def test_normalisation(self):
        returns = _discount_returns([10.0, 20.0, 5.0], gamma=0.99)
        # Should be normalised to [-1, 1]
        for r in returns:
            assert -1.0 <= r <= 1.0

    def test_zero_rewards(self):
        returns = _discount_returns([0.0, 0.0, 0.0], gamma=0.99)
        for r in returns:
            assert r == 0.0

    def test_negative_rewards(self):
        returns = _discount_returns([-1.0, -1.0], gamma=0.99)
        assert len(returns) == 2
        for r in returns:
            assert -1.0 <= r <= 1.0

    def test_gamma_zero(self):
        returns = _discount_returns([1.0, 2.0, 3.0], gamma=0.0)
        # With gamma=0, each return is just the immediate reward (normalised)
        assert len(returns) == 3
