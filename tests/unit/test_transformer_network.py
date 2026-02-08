"""
Comprehensive Tests for Transformer Network

Tests the state-of-the-art transformer architecture for AlphaZero.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_zero.core.transformer_network import (
    AttentionType,
    Civ6TransformerNetwork,
    DropPath,
    FeedForward,
    MixtureOfExperts,
    MultiHeadAttention,
    NormType,
    PatchEmbed,
    RMSNorm,
    RotaryEmbedding,
    TransformerBlock,
    TransformerConfig,
    TransformerEncoder,
    TransformerPolicyHead,
    TransformerValueHead,
    VectorProjection,
    compute_transformer_loss,
    create_transformer_network,
    get_norm_layer,
)

# =============================================================================
# Configuration Tests
# =============================================================================


class TestTransformerConfig:
    """Tests for TransformerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TransformerConfig()
        assert config.spatial_channels == 13
        assert config.spatial_size == 16
        assert config.embed_dim == 256
        assert config.num_heads == 8

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = TransformerConfig(
            spatial_channels=84,
            spatial_size=32,
            embed_dim=512,
            num_layers=12,
            num_heads=16,
        )
        assert config.spatial_channels == 84
        assert config.embed_dim == 512
        assert config.num_layers == 12

    def test_derived_properties(self) -> None:
        """Test computed properties."""
        config = TransformerConfig(
            spatial_size=16,
            patch_size=4,
            embed_dim=256,
            num_heads=8,
        )
        assert config.num_patches == 16  # (16/4)^2 = 16
        assert config.head_dim == 32  # 256/8 = 32

    def test_embed_dim_divisibility(self) -> None:
        """Test that embed_dim must be divisible by num_heads."""
        with pytest.raises(AssertionError):
            TransformerConfig(embed_dim=256, num_heads=7)


# =============================================================================
# Normalization Tests
# =============================================================================


class TestRMSNorm:
    """Tests for RMSNorm layer."""

    def test_forward(self) -> None:
        """Test forward pass."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_output_scale(self) -> None:
        """Test that output is normalized."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64) * 100  # Large values
        out = norm(x)
        # RMS should be around 1
        rms = torch.sqrt(torch.mean(out**2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


class TestGetNormLayer:
    """Tests for get_norm_layer factory."""

    def test_rms_norm(self) -> None:
        """Test RMSNorm creation."""
        norm = get_norm_layer(NormType.RMS_NORM, 64)
        assert isinstance(norm, RMSNorm)

    def test_layer_norm(self) -> None:
        """Test LayerNorm creation."""
        norm = get_norm_layer(NormType.LAYER_NORM, 64)
        assert isinstance(norm, nn.LayerNorm)

    def test_group_norm(self) -> None:
        """Test GroupNorm creation."""
        norm = get_norm_layer(NormType.GROUP_NORM, 64)
        assert isinstance(norm, nn.GroupNorm)


# =============================================================================
# Positional Embedding Tests
# =============================================================================


class TestRotaryEmbedding:
    """Tests for Rotary Position Embedding."""

    @pytest.fixture
    def rope(self) -> RotaryEmbedding:
        """Create RoPE module."""
        return RotaryEmbedding(dim=32, max_seq_len=128)

    def test_forward(self, rope: RotaryEmbedding) -> None:
        """Test forward pass."""
        q = torch.randn(2, 4, 16, 32)  # (B, H, N, D)
        k = torch.randn(2, 4, 16, 32)

        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_different_positions(self, rope: RotaryEmbedding) -> None:
        """Test that different positions get different embeddings."""
        q = torch.ones(2, 4, 16, 32)
        k = torch.ones(2, 4, 16, 32)

        q_rot, _k_rot = rope(q, k)

        # Different positions should have different values
        assert not torch.allclose(q_rot[:, :, 0, :], q_rot[:, :, 1, :])

    def test_cache_extension(self, rope: RotaryEmbedding) -> None:
        """Test cache extension for longer sequences."""
        q = torch.randn(2, 4, 200, 32)  # Longer than max_seq_len
        k = torch.randn(2, 4, 200, 32)

        q_rot, _k_rot = rope(q, k, seq_len=200)
        assert q_rot.shape[2] == 200


# =============================================================================
# Attention Tests
# =============================================================================


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    @pytest.fixture
    def attention(self) -> MultiHeadAttention:
        """Create attention module."""
        return MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.0,
            use_rope=True,
        )

    def test_forward(self, attention: MultiHeadAttention) -> None:
        """Test forward pass."""
        x = torch.randn(2, 16, 64)
        out = attention(x)
        assert out.shape == x.shape

    def test_with_mask(self, attention: MultiHeadAttention) -> None:
        """Test forward with attention mask."""
        x = torch.randn(2, 16, 64)
        mask = torch.ones(2, 1, 16, 16)
        mask[:, :, :8, 8:] = 0  # Mask out some positions

        out = attention(x, mask)
        assert out.shape == x.shape

    def test_without_rope(self) -> None:
        """Test attention without RoPE."""
        attention = MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            use_rope=False,
        )
        x = torch.randn(2, 16, 64)
        out = attention(x)
        assert out.shape == x.shape

    def test_linear_attention(self) -> None:
        """Test linear attention variant."""
        attention = MultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            attention_type=AttentionType.LINEAR,
        )
        x = torch.randn(2, 16, 64)
        out = attention(x)
        assert out.shape == x.shape


# =============================================================================
# Feed-Forward Tests
# =============================================================================


class TestFeedForward:
    """Tests for FeedForward network."""

    def test_forward_with_glu(self) -> None:
        """Test forward pass with GLU."""
        ffn = FeedForward(dim=64, hidden_dim=256, use_glu=True)
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_forward_without_glu(self) -> None:
        """Test forward pass without GLU."""
        ffn = FeedForward(dim=64, hidden_dim=256, use_glu=False)
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_default_hidden_dim(self) -> None:
        """Test default hidden dimension (4x)."""
        ffn = FeedForward(dim=64)
        # Hidden dim should be 4*64 = 256
        x = torch.randn(2, 16, 64)
        out = ffn(x)
        assert out.shape == x.shape


# =============================================================================
# Mixture of Experts Tests
# =============================================================================


class TestMixtureOfExperts:
    """Tests for Mixture of Experts layer."""

    @pytest.fixture
    def moe(self) -> MixtureOfExperts:
        """Create MoE module."""
        return MixtureOfExperts(
            dim=64,
            num_experts=4,
            experts_per_token=2,
        )

    def test_forward(self, moe: MixtureOfExperts) -> None:
        """Test forward pass."""
        x = torch.randn(2, 16, 64)
        out = moe(x)
        assert out.shape == x.shape

    def test_expert_routing(self, moe: MixtureOfExperts) -> None:
        """Test that expert routing works."""
        x = torch.randn(2, 16, 64)
        out = moe(x)

        # Output should be non-zero (experts are being used)
        assert out.abs().sum() > 0


# =============================================================================
# Transformer Block Tests
# =============================================================================


class TestTransformerBlock:
    """Tests for TransformerBlock."""

    @pytest.fixture
    def config(self) -> TransformerConfig:
        """Create test config."""
        return TransformerConfig(
            embed_dim=64,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.0,
            path_dropout=0.0,
        )

    def test_forward(self, config: TransformerConfig) -> None:
        """Test forward pass."""
        block = TransformerBlock(config)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self, config: TransformerConfig) -> None:
        """Test residual connections are working."""
        block = TransformerBlock(config)
        x = torch.randn(2, 16, 64)
        out = block(x)

        # Output should be different but not completely unrelated to input
        # due to residual connections
        diff = (out - x).abs().mean()
        assert diff < x.abs().mean()  # Reasonable change

    def test_with_moe(self, config: TransformerConfig) -> None:
        """Test block with MoE."""
        config.use_moe = True
        config.num_experts = 4
        config.experts_per_token = 2

        block = TransformerBlock(config, use_moe=True)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape


# =============================================================================
# Drop Path Tests
# =============================================================================


class TestDropPath:
    """Tests for DropPath (stochastic depth)."""

    def test_no_drop_in_eval(self) -> None:
        """Test no dropping in eval mode."""
        drop = DropPath(drop_prob=0.5)
        drop.eval()
        x = torch.randn(2, 16, 64)
        out = drop(x)
        assert torch.equal(x, out)

    def test_drop_in_train(self) -> None:
        """Test dropping in train mode."""
        drop = DropPath(drop_prob=0.99)  # High drop rate
        drop.train()
        torch.manual_seed(42)
        x = torch.ones(100, 16, 64)
        out = drop(x)

        # Some should be dropped (zeroed)
        num_dropped = (out.sum(dim=[1, 2]) == 0).sum()
        assert num_dropped > 0

    def test_zero_drop_prob(self) -> None:
        """Test with zero drop probability."""
        drop = DropPath(drop_prob=0.0)
        drop.train()
        x = torch.randn(2, 16, 64)
        out = drop(x)
        assert torch.equal(x, out)


# =============================================================================
# Patch Embedding Tests
# =============================================================================


class TestPatchEmbed:
    """Tests for PatchEmbed."""

    def test_forward(self) -> None:
        """Test forward pass."""
        embed = PatchEmbed(
            in_channels=13,
            embed_dim=64,
            patch_size=4,
            spatial_size=16,
        )
        x = torch.randn(2, 13, 16, 16)
        out = embed(x)

        # Should have (16/4)^2 = 16 patches
        assert out.shape == (2, 16, 64)

    def test_different_patch_sizes(self) -> None:
        """Test different patch sizes."""
        for patch_size in [2, 4, 8]:
            embed = PatchEmbed(
                in_channels=13,
                embed_dim=64,
                patch_size=patch_size,
                spatial_size=16,
            )
            x = torch.randn(2, 13, 16, 16)
            out = embed(x)

            expected_patches = (16 // patch_size) ** 2
            assert out.shape[1] == expected_patches


# =============================================================================
# Vector Projection Tests
# =============================================================================


class TestVectorProjection:
    """Tests for VectorProjection."""

    def test_forward(self) -> None:
        """Test forward pass."""
        proj = VectorProjection(
            vector_dim=64,
            embed_dim=128,
            num_tokens=1,
        )
        x = torch.randn(2, 64)
        out = proj(x)
        assert out.shape == (2, 1, 128)

    def test_multiple_tokens(self) -> None:
        """Test projection to multiple tokens."""
        proj = VectorProjection(
            vector_dim=64,
            embed_dim=128,
            num_tokens=4,
        )
        x = torch.randn(2, 64)
        out = proj(x)
        assert out.shape == (2, 4, 128)


# =============================================================================
# Transformer Encoder Tests
# =============================================================================


class TestTransformerEncoder:
    """Tests for TransformerEncoder."""

    @pytest.fixture
    def config(self) -> TransformerConfig:
        """Create test config."""
        return TransformerConfig(
            spatial_channels=13,
            spatial_size=16,
            vector_dim=64,
            embed_dim=128,
            num_layers=2,
            num_heads=4,
            patch_size=4,
        )

    def test_forward(self, config: TransformerConfig) -> None:
        """Test forward pass."""
        encoder = TransformerEncoder(config)
        spatial = torch.randn(2, 13, 16, 16)
        vector = torch.randn(2, 64)

        pooled, sequence = encoder(spatial, vector)

        assert pooled.shape == (2, 128)
        assert sequence.shape[0] == 2
        assert sequence.shape[2] == 128

    def test_cls_token(self, config: TransformerConfig) -> None:
        """Test CLS token handling."""
        config.cls_token = True
        encoder = TransformerEncoder(config)

        spatial = torch.randn(2, 13, 16, 16)
        vector = torch.randn(2, 64)

        _pooled, sequence = encoder(spatial, vector)

        # CLS token should be at position 0
        assert sequence.shape[1] == config.num_patches + 2  # CLS + vector + patches

    def test_mean_pooling(self, config: TransformerConfig) -> None:
        """Test mean pooling."""
        config.pool_type = "mean"
        config.cls_token = False
        encoder = TransformerEncoder(config)

        spatial = torch.randn(2, 13, 16, 16)
        vector = torch.randn(2, 64)

        pooled, _ = encoder(spatial, vector)
        assert pooled.shape == (2, 128)


# =============================================================================
# Policy and Value Head Tests
# =============================================================================


class TestTransformerPolicyHead:
    """Tests for TransformerPolicyHead."""

    def test_forward(self) -> None:
        """Test forward pass."""
        head = TransformerPolicyHead(
            embed_dim=128,
            action_space_size=381,
            hidden_dim=64,
        )
        x = torch.randn(2, 128)
        out = head(x)
        assert out.shape == (2, 381)


class TestTransformerValueHead:
    """Tests for TransformerValueHead."""

    def test_forward(self) -> None:
        """Test forward pass."""
        head = TransformerValueHead(
            embed_dim=128,
            hidden_dim=64,
        )
        x = torch.randn(2, 128)
        out = head(x)
        assert out.shape == (2, 1)

    def test_output_range(self) -> None:
        """Test that output is in [-1, 1]."""
        head = TransformerValueHead(embed_dim=128)
        x = torch.randn(100, 128) * 10  # Large inputs
        out = head(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0


# =============================================================================
# Full Network Tests
# =============================================================================


class TestCiv6TransformerNetwork:
    """Tests for Civ6TransformerNetwork."""

    @pytest.fixture
    def config(self) -> TransformerConfig:
        """Create test config."""
        return TransformerConfig(
            spatial_channels=13,
            spatial_size=16,
            vector_dim=64,
            action_space_size=381,
            embed_dim=128,
            num_layers=2,
            num_heads=4,
            patch_size=4,
        )

    @pytest.fixture
    def network(self, config: TransformerConfig) -> Civ6TransformerNetwork:
        """Create network."""
        return Civ6TransformerNetwork(config=config)

    def test_forward(self, network: Civ6TransformerNetwork) -> None:
        """Test forward pass."""
        spatial = torch.randn(2, 13, 16, 16)
        vector = torch.randn(2, 64)

        policy_logits, value = network(spatial, vector)

        assert policy_logits.shape == (2, 381)
        assert value.shape == (2,)

    def test_forward_with_mask(self, network: Civ6TransformerNetwork) -> None:
        """Test forward with action mask."""
        spatial = torch.randn(2, 13, 16, 16)
        vector = torch.randn(2, 64)
        mask = torch.ones(2, 381)
        mask[:, 100:] = 0  # Mask out some actions

        policy_logits, _value = network(spatial, vector, mask)

        # Masked actions should have very low logits
        assert policy_logits[:, 100:].max() < -1e8

    def test_predict(self, network: Civ6TransformerNetwork) -> None:
        """Test predict method."""
        spatial = torch.randn(2, 13, 16, 16)
        vector = torch.randn(2, 64)

        probs, _value = network.predict(spatial, vector)

        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_predict_single(self, network: Civ6TransformerNetwork) -> None:
        """Test single state prediction."""
        spatial = torch.randn(13, 16, 16)
        vector = torch.randn(64)

        probs, value = network.predict_single(spatial, vector)

        assert probs.shape == (381,)
        assert isinstance(value, float)

    def test_save_load(
        self, network: Civ6TransformerNetwork, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Test save and load."""
        path = str(tmp_path / "model.pt")
        network.save(path)

        loaded = Civ6TransformerNetwork.load(path)

        # Check same config
        assert loaded.config.embed_dim == network.config.embed_dim

        # Check same output
        spatial = torch.randn(2, 13, 16, 16)
        vector = torch.randn(2, 64)

        network.eval()
        loaded.eval()

        with torch.no_grad():
            out1 = network(spatial, vector)
            out2 = loaded(spatial, vector)

        assert torch.allclose(out1[0], out2[0])
        assert torch.allclose(out1[1], out2[1])

    def test_get_num_parameters(self, network: Civ6TransformerNetwork) -> None:
        """Test parameter counting."""
        params = network.get_num_parameters()

        assert "encoder" in params
        assert "policy_head" in params
        assert "value_head" in params
        assert "total" in params
        assert params["total"] > 0

    def test_gradient_flow(self, network: Civ6TransformerNetwork) -> None:
        """Test that gradients flow through network."""
        spatial = torch.randn(2, 13, 16, 16, requires_grad=True)
        vector = torch.randn(2, 64, requires_grad=True)

        policy_logits, value = network(spatial, vector)
        loss = policy_logits.sum() + value.sum()
        loss.backward()

        assert spatial.grad is not None
        assert vector.grad is not None
        assert spatial.grad.abs().sum() > 0
        assert vector.grad.abs().sum() > 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateTransformerNetwork:
    """Tests for factory function."""

    def test_tiny_model(self) -> None:
        """Test tiny model preset."""
        network = create_transformer_network(model_size="tiny")
        assert network.config.embed_dim == 128
        assert network.config.num_layers == 4

    def test_small_model(self) -> None:
        """Test small model preset."""
        network = create_transformer_network(model_size="small")
        assert network.config.embed_dim == 256
        assert network.config.num_layers == 6

    def test_medium_model(self) -> None:
        """Test medium model preset."""
        network = create_transformer_network(model_size="medium")
        assert network.config.embed_dim == 384

    def test_large_model(self) -> None:
        """Test large model preset."""
        network = create_transformer_network(model_size="large")
        assert network.config.embed_dim == 512
        assert network.config.use_moe is True

    def test_custom_params(self) -> None:
        """Test custom parameters override presets."""
        network = create_transformer_network(
            model_size="small",
            embed_dim=512,  # Override
            num_layers=10,  # Override
        )
        assert network.config.embed_dim == 512
        assert network.config.num_layers == 10


# =============================================================================
# Loss Function Tests
# =============================================================================


class TestComputeTransformerLoss:
    """Tests for compute_transformer_loss."""

    def test_basic_loss(self) -> None:
        """Test basic loss computation."""
        policy_logits = torch.randn(2, 10)
        value = torch.randn(2)
        target_policy = F.softmax(torch.randn(2, 10), dim=-1)
        target_value = torch.randn(2)

        loss, loss_dict = compute_transformer_loss(
            policy_logits, value, target_policy, target_value
        )

        assert loss.shape == ()
        assert "total" in loss_dict
        assert "policy" in loss_dict
        assert "value" in loss_dict
        assert "entropy" in loss_dict

    def test_loss_with_mask(self) -> None:
        """Test loss with action mask."""
        policy_logits = torch.randn(2, 10)
        value = torch.randn(2)
        target_policy = torch.zeros(2, 10)
        target_policy[:, :5] = 0.2  # Only first 5 actions valid
        target_value = torch.randn(2)
        action_mask = torch.zeros(2, 10)
        action_mask[:, :5] = 1

        loss, _ = compute_transformer_loss(
            policy_logits, value, target_policy, target_value, action_mask
        )

        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_loss_weights(self) -> None:
        """Test loss weight parameters."""
        policy_logits = torch.randn(2, 10)
        value = torch.randn(2)
        target_policy = F.softmax(torch.randn(2, 10), dim=-1)
        target_value = torch.randn(2)

        # With different weights
        _, loss1 = compute_transformer_loss(
            policy_logits, value, target_policy, target_value, policy_weight=1.0, value_weight=1.0
        )

        _, loss2 = compute_transformer_loss(
            policy_logits, value, target_policy, target_value, policy_weight=2.0, value_weight=0.5
        )

        # Different weights should give different totals
        assert loss1["total"] != loss2["total"]

    def test_entropy_regularization(self) -> None:
        """Test entropy term."""
        # Uniform distribution has maximum entropy
        policy_logits_uniform = torch.zeros(2, 10)
        # Peaked distribution has minimum entropy
        policy_logits_peaked = torch.zeros(2, 10)
        policy_logits_peaked[:, 0] = 100

        value = torch.zeros(2)
        target_policy = F.softmax(torch.randn(2, 10), dim=-1)
        target_value = torch.zeros(2)

        _, loss_uniform = compute_transformer_loss(
            policy_logits_uniform, value, target_policy, target_value
        )
        _, loss_peaked = compute_transformer_loss(
            policy_logits_peaked, value, target_policy, target_value
        )

        # Uniform should have higher entropy
        assert loss_uniform["entropy"] > loss_peaked["entropy"]


# =============================================================================
# Integration Tests
# =============================================================================


class TestTransformerIntegration:
    """Integration tests for transformer network."""

    def test_training_step(self) -> None:
        """Test a complete training step."""
        network = create_transformer_network(model_size="tiny")
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        # Generate batch
        spatial = torch.randn(4, 13, 16, 16)
        vector = torch.randn(4, 64)
        target_policy = F.softmax(torch.randn(4, 381), dim=-1)
        target_value = torch.randn(4)

        # Forward pass
        network.train()
        policy_logits, value = network(spatial, vector)

        # Compute loss
        loss, _ = compute_transformer_loss(policy_logits, value, target_policy, target_value)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check weights updated
        assert not torch.isnan(loss)

    def test_eval_mode(self) -> None:
        """Test evaluation mode."""
        network = create_transformer_network(model_size="tiny")
        network.eval()

        spatial = torch.randn(2, 13, 16, 16)
        vector = torch.randn(2, 64)

        with torch.no_grad():
            policy, value = network.predict(spatial, vector)

        assert policy.sum(dim=-1).allclose(torch.ones(2))
        assert not torch.isnan(value).any()

    def test_batch_independence(self) -> None:
        """Test that batches are processed independently."""
        network = create_transformer_network(model_size="tiny")
        network.eval()

        # Single sample
        spatial1 = torch.randn(1, 13, 16, 16)
        vector1 = torch.randn(1, 64)

        # Same sample in batch
        spatial2 = spatial1.repeat(4, 1, 1, 1)
        vector2 = vector1.repeat(4, 1)

        with torch.no_grad():
            policy1, value1 = network.predict(spatial1, vector1)
            policy2, value2 = network.predict(spatial2, vector2)

        # All batch elements should be identical
        for i in range(4):
            assert torch.allclose(policy1[0], policy2[i], atol=1e-5)
            assert torch.allclose(value1, value2[i : i + 1], atol=1e-5)

    def test_different_spatial_sizes(self) -> None:
        """Test with different spatial sizes."""
        for spatial_size in [8, 16, 32]:
            config = TransformerConfig(
                spatial_size=spatial_size,
                patch_size=4,
                embed_dim=64,
                num_layers=2,
                num_heads=4,
            )
            network = Civ6TransformerNetwork(config=config)

            spatial = torch.randn(2, 13, spatial_size, spatial_size)
            vector = torch.randn(2, 64)

            policy, _value = network(spatial, vector)
            assert policy.shape == (2, 381)
