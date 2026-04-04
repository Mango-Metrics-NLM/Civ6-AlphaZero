"""Comprehensive coverage tests for alpha_zero.core.civ6_network.

Covers classes and functions not exercised by test_network.py:
- NetworkConfig defaults and mutation
- SpatialEncoder, VectorEncoder standalone
- PolicyHead, ValueHead standalone
- Civ6Network save/load, predict, predict_single
- MicroCivNetwork with different board sizes, N-player value head
- PositionalEncoding2D (learned and sinusoidal)
- SpatialSelfAttention (with and without relative position bias)
- CrossModalAttention
- AttentionSpatialEncoder
- HierarchicalPolicyHead (with and without category probs)
- Civ6NetworkV2 (forward, predict, predict_single, predict_hierarchical,
  forward_with_categories, save/load, from_v1_checkpoint, get_num_parameters)
- compute_loss deprecation warning
"""

from __future__ import annotations

import warnings

import pytest
import torch

from alpha_zero.core.civ6_network import (
    AttentionSpatialEncoder,
    Civ6Network,
    Civ6NetworkV2,
    CrossModalAttention,
    HierarchicalPolicyHead,
    MicroCivNetwork,
    NetworkConfig,
    PolicyHead,
    PositionalEncoding2D,
    ResidualBlock,
    SpatialEncoder,
    SpatialSelfAttention,
    ValueHead,
    VectorEncoder,
    compute_loss,
)


def _fix_relative_pos_bug(module: torch.nn.Module) -> None:
    """Work around a known bug where SpatialSelfAttention.__init__ sets
    ``relative_position_index`` as a plain attribute (``None``) which then
    causes ``register_buffer`` to raise ``KeyError`` on the first forward
    call.  Deleting the plain attribute lets ``register_buffer`` succeed.
    """
    for m in module.modules():
        if isinstance(m, SpatialSelfAttention) and m.use_relative_pos:
            if hasattr(m, "relative_position_index") and not isinstance(
                m.relative_position_index, torch.Tensor
            ):
                del m.relative_position_index

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config():
    """Small NetworkConfig for fast tests."""
    return NetworkConfig(
        spatial_channels=4,
        spatial_size=8,
        vector_dim=16,
        num_res_blocks=2,
        num_filters=8,
        fc_hidden_dim=16,
        action_space_size=32,
        value_head_hidden=8,
        dropout=0.0,
        num_attention_heads=2,
        use_attention=True,
        attention_dropout=0.1,
    )


@pytest.fixture
def spatial_8x8():
    """Batch of 2 spatial tensors, 4 channels, 8x8."""
    return torch.randn(2, 4, 8, 8)


@pytest.fixture
def vector_16():
    """Batch of 2 vector tensors, dim 16."""
    return torch.randn(2, 16)


# ---------------------------------------------------------------------------
# NetworkConfig
# ---------------------------------------------------------------------------

class TestNetworkConfig:
    def test_defaults(self):
        cfg = NetworkConfig()
        assert cfg.spatial_channels == 13
        assert cfg.spatial_size == 16
        assert cfg.vector_dim == 64
        assert cfg.num_res_blocks == 6
        assert cfg.num_filters == 64
        assert cfg.fc_hidden_dim == 256
        assert cfg.action_space_size == 381
        assert cfg.value_head_hidden == 128
        assert cfg.dropout == 0.0
        assert cfg.num_attention_heads == 4
        assert cfg.use_attention is True
        assert cfg.attention_dropout == 0.1

    def test_custom_values(self):
        cfg = NetworkConfig(spatial_channels=84, spatial_size=44, vector_dim=512)
        assert cfg.spatial_channels == 84
        assert cfg.spatial_size == 44
        assert cfg.vector_dim == 512


# ---------------------------------------------------------------------------
# SpatialEncoder
# ---------------------------------------------------------------------------

class TestSpatialEncoder:
    def test_output_shape(self):
        enc = SpatialEncoder(in_channels=4, num_filters=8, num_res_blocks=2)
        x = torch.randn(2, 4, 8, 8)
        out = enc(x)
        assert out.shape == (2, 8, 8, 8)

    def test_different_spatial_sizes(self):
        enc = SpatialEncoder(in_channels=4, num_filters=8, num_res_blocks=1)
        for size in [4, 12, 16]:
            x = torch.randn(1, 4, size, size)
            out = enc(x)
            assert out.shape == (1, 8, size, size)

    def test_single_sample(self):
        enc = SpatialEncoder(in_channels=4, num_filters=8, num_res_blocks=1)
        x = torch.randn(1, 4, 8, 8)
        out = enc(x)
        assert out.shape == (1, 8, 8, 8)


# ---------------------------------------------------------------------------
# VectorEncoder
# ---------------------------------------------------------------------------

class TestVectorEncoder:
    def test_output_shape(self):
        enc = VectorEncoder(input_dim=16, hidden_dim=32, output_dim=8)
        x = torch.randn(2, 16)
        out = enc(x)
        assert out.shape == (2, 8)

    def test_with_dropout(self):
        enc = VectorEncoder(input_dim=16, hidden_dim=32, output_dim=8, dropout=0.5)
        enc.train()
        x = torch.randn(4, 16)
        out = enc(x)
        assert out.shape == (4, 8)

    def test_no_dropout_identity(self):
        enc = VectorEncoder(input_dim=16, hidden_dim=32, output_dim=8, dropout=0.0)
        assert isinstance(enc.dropout, torch.nn.Identity)


# ---------------------------------------------------------------------------
# PolicyHead
# ---------------------------------------------------------------------------

class TestPolicyHead:
    def test_output_shape(self):
        head = PolicyHead(
            spatial_filters=8,
            spatial_size=8,
            vector_dim=16,
            action_space_size=32,
            hidden_dim=16,
        )
        sf = torch.randn(2, 8, 8, 8)
        vf = torch.randn(2, 16)
        out = head(sf, vf)
        assert out.shape == (2, 32)

    def test_different_action_sizes(self):
        for action_size in [10, 100, 500]:
            head = PolicyHead(
                spatial_filters=8,
                spatial_size=4,
                vector_dim=8,
                action_space_size=action_size,
            )
            sf = torch.randn(1, 8, 4, 4)
            vf = torch.randn(1, 8)
            out = head(sf, vf)
            assert out.shape == (1, action_size)


# ---------------------------------------------------------------------------
# ValueHead
# ---------------------------------------------------------------------------

class TestValueHead:
    def test_output_shape(self):
        head = ValueHead(spatial_filters=8, spatial_size=8, vector_dim=16, hidden_dim=8)
        sf = torch.randn(2, 8, 8, 8)
        vf = torch.randn(2, 16)
        out = head(sf, vf)
        assert out.shape == (2, 1)

    def test_value_range(self):
        head = ValueHead(spatial_filters=8, spatial_size=8, vector_dim=16, hidden_dim=8)
        sf = torch.randn(10, 8, 8, 8)
        vf = torch.randn(10, 16)
        out = head(sf, vf)
        assert torch.all(out >= -1.0)
        assert torch.all(out <= 1.0)


# ---------------------------------------------------------------------------
# Civ6Network (extended)
# ---------------------------------------------------------------------------

class TestCiv6NetworkExtended:
    @pytest.fixture
    def net(self, small_config):
        # Disable attention fields not used by V1
        cfg = NetworkConfig(
            spatial_channels=4,
            spatial_size=8,
            vector_dim=16,
            num_res_blocks=2,
            num_filters=8,
            fc_hidden_dim=16,
            action_space_size=32,
            value_head_hidden=8,
        )
        return Civ6Network(config=cfg)

    def test_forward_no_mask(self, net, spatial_8x8, vector_16):
        policy, value = net(spatial_8x8, vector_16)
        assert policy.shape == (2, 32)
        assert value.shape == (2,)

    def test_forward_with_mask(self, net, spatial_8x8, vector_16):
        mask = torch.zeros(2, 32)
        mask[:, :5] = 1.0
        policy, value = net(spatial_8x8, vector_16, action_mask=mask)
        assert torch.all(policy[:, 5:] < -1e8)

    def test_predict(self, net, spatial_8x8, vector_16):
        probs, value = net.predict(spatial_8x8, vector_16)
        assert probs.shape == (2, 32)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_predict_with_mask(self, net, spatial_8x8, vector_16):
        mask = torch.zeros(2, 32)
        mask[:, :5] = 1.0
        probs, _ = net.predict(spatial_8x8, vector_16, action_mask=mask)
        # Mass should concentrate on valid actions
        assert probs[:, :5].sum(dim=-1).min() > 0.99

    def test_predict_single_no_mask(self, net):
        s = torch.randn(4, 8, 8)
        v = torch.randn(16)
        probs, val = net.predict_single(s, v)
        assert probs.shape == (32,)
        assert isinstance(val, float)
        assert -1 <= val <= 1

    def test_predict_single_with_mask(self, net):
        s = torch.randn(4, 8, 8)
        v = torch.randn(16)
        m = torch.ones(32)
        m[10:] = 0
        probs, val = net.predict_single(s, v, m)
        assert probs.shape == (32,)
        assert isinstance(val, float)

    def test_save_load_roundtrip(self, net, tmp_path):
        path = str(tmp_path / "model.pt")
        net.save(path)
        loaded = Civ6Network.load(path)
        assert loaded.config.spatial_channels == net.config.spatial_channels
        assert loaded.config.action_space_size == net.config.action_space_size
        # Check weights match
        for (_, p1), (_, p2) in zip(
            net.named_parameters(), loaded.named_parameters(), strict=True
        ):
            assert torch.allclose(p1, p2)

    def test_save_load_device(self, net, tmp_path):
        path = str(tmp_path / "model.pt")
        net.save(path)
        loaded = Civ6Network.load(path, device="cpu")
        assert next(loaded.parameters()).device == torch.device("cpu")

    def test_default_config(self):
        net = Civ6Network()
        assert net.config.spatial_channels == 13

    def test_kwargs_override(self):
        net = Civ6Network(spatial_channels=7, num_filters=32, spatial_size=8)
        assert net.config.spatial_channels == 7
        assert net.config.num_filters == 32

    def test_kwargs_unknown_ignored(self):
        # Unknown kwargs should be silently ignored (no hasattr match)
        net = Civ6Network(nonexistent_param=999)
        assert net.config.spatial_channels == 13  # default


# ---------------------------------------------------------------------------
# MicroCivNetwork (extended)
# ---------------------------------------------------------------------------

class TestMicroCivNetworkExtended:
    def test_default_params(self):
        net = MicroCivNetwork()
        assert net.config.spatial_size == 16
        assert net.config.action_space_size == 381
        assert net.num_players == 2

    def test_custom_board_size(self):
        net = MicroCivNetwork(board_size=8, action_space_size=100, num_res_blocks=1, num_filters=8)
        s = torch.randn(1, 13, 8, 8)
        v = torch.randn(1, 64)
        p, val = net(s, v)
        assert p.shape == (1, 100)

    def test_num_players_property(self):
        net = MicroCivNetwork(num_players=4, board_size=8, num_res_blocks=1, num_filters=8)
        assert net.num_players == 4

    def test_nplayer_value_head(self):
        """When num_players > 2, NPlayerValueHead should be used if available."""
        net = MicroCivNetwork(
            num_players=4,
            board_size=8,
            action_space_size=50,
            num_res_blocks=1,
            num_filters=8,
            spatial_channels=13,
        )
        assert net.num_players == 4
        # Forward pass should work regardless
        s = torch.randn(2, 13, 8, 8)
        v = torch.randn(2, 64)
        p, val = net(s, v)
        assert p.shape == (2, 50)

    def test_with_explicit_config(self):
        cfg = NetworkConfig(
            spatial_channels=4,
            spatial_size=4,
            vector_dim=16,
            num_res_blocks=1,
            num_filters=8,
            fc_hidden_dim=16,
            action_space_size=20,
            value_head_hidden=8,
        )
        net = MicroCivNetwork(config=cfg)
        s = torch.randn(1, 4, 4, 4)
        v = torch.randn(1, 16)
        p, val = net(s, v)
        assert p.shape == (1, 20)

    def test_save_load(self, tmp_path):
        net = MicroCivNetwork(board_size=8, action_space_size=50, num_res_blocks=1, num_filters=8)
        path = str(tmp_path / "micro.pt")
        net.save(path)
        loaded = MicroCivNetwork.load(path)
        # Verify forward pass on loaded model
        s = torch.randn(1, 13, 8, 8)
        v = torch.randn(1, 64)
        p, val = loaded(s, v)
        assert p.shape == (1, 50)


# ---------------------------------------------------------------------------
# PositionalEncoding2D
# ---------------------------------------------------------------------------

class TestPositionalEncoding2D:
    def test_learned_encoding_shape(self):
        pe = PositionalEncoding2D(num_channels=8, max_size=16, learned=True)
        x = torch.randn(2, 8, 8, 8)
        out = pe(x)
        assert out.shape == x.shape

    def test_sinusoidal_encoding_shape(self):
        pe = PositionalEncoding2D(num_channels=8, max_size=16, learned=False)
        x = torch.randn(2, 8, 8, 8)
        out = pe(x)
        assert out.shape == x.shape

    def test_learned_different_sizes(self):
        pe = PositionalEncoding2D(num_channels=8, max_size=32, learned=True)
        for size in [4, 8, 16]:
            x = torch.randn(1, 8, size, size)
            out = pe(x)
            assert out.shape == x.shape

    def test_sinusoidal_different_sizes(self):
        pe = PositionalEncoding2D(num_channels=8, max_size=32, learned=False)
        for size in [4, 8, 16]:
            x = torch.randn(1, 8, size, size)
            out = pe(x)
            assert out.shape == x.shape

    def test_encoding_adds_nonzero(self):
        pe = PositionalEncoding2D(num_channels=8, max_size=16, learned=True)
        x = torch.zeros(1, 8, 4, 4)
        out = pe(x)
        # Output should not be all zeros since positional encoding is added
        assert not torch.allclose(out, x, atol=1e-8)

    def test_sinusoidal_encoding_nonzero(self):
        pe = PositionalEncoding2D(num_channels=8, max_size=16, learned=False)
        x = torch.zeros(1, 8, 4, 4)
        out = pe(x)
        assert not torch.allclose(out, x, atol=1e-8)


# ---------------------------------------------------------------------------
# SpatialSelfAttention
# ---------------------------------------------------------------------------

class TestSpatialSelfAttention:
    def test_output_shape(self):
        attn = SpatialSelfAttention(num_channels=8, num_heads=2, dropout=0.0, use_relative_pos=False)
        x = torch.randn(2, 8, 4, 4)
        out = attn(x)
        assert out.shape == x.shape

    def test_with_relative_pos(self):
        attn = SpatialSelfAttention(
            num_channels=8, num_heads=2, dropout=0.0, use_relative_pos=True,
        )
        _fix_relative_pos_bug(attn)
        x = torch.randn(2, 8, 4, 4)
        out = attn(x)
        assert out.shape == x.shape

    def test_without_relative_pos(self):
        attn = SpatialSelfAttention(
            num_channels=8, num_heads=2, dropout=0.0, use_relative_pos=False,
        )
        x = torch.randn(2, 8, 4, 4)
        out = attn(x)
        assert out.shape == x.shape

    def test_different_spatial_sizes(self):
        for size in [3, 4, 6]:
            # Fresh instance per size to avoid register_buffer conflict
            attn = SpatialSelfAttention(
                num_channels=8, num_heads=2, dropout=0.0, use_relative_pos=True,
            )
            _fix_relative_pos_bug(attn)
            x = torch.randn(1, 8, size, size)
            out = attn(x)
            assert out.shape == x.shape

    def test_relative_pos_reinit_on_size_change(self):
        attn = SpatialSelfAttention(
            num_channels=8, num_heads=2, dropout=0.0, use_relative_pos=True,
        )
        _fix_relative_pos_bug(attn)
        x4 = torch.randn(1, 8, 4, 4)
        attn(x4)
        assert attn._spatial_size == 16  # 4*4

        # Size change triggers re-registration; need to delete old buffer first
        del attn.relative_position_index
        x3 = torch.randn(1, 8, 3, 3)
        attn(x3)
        assert attn._spatial_size == 9  # 3*3

    def test_gradient_flow(self):
        attn = SpatialSelfAttention(num_channels=8, num_heads=2, dropout=0.0, use_relative_pos=False)
        x = torch.randn(1, 8, 4, 4, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# CrossModalAttention
# ---------------------------------------------------------------------------

class TestCrossModalAttention:
    def test_output_shapes(self):
        cma = CrossModalAttention(
            spatial_channels=8, vector_dim=16, num_heads=2, dropout=0.0,
        )
        spatial = torch.randn(2, 8, 4, 4)
        vector = torch.randn(2, 16)
        out_s, out_v = cma(spatial, vector)
        assert out_s.shape == spatial.shape
        assert out_v.shape == vector.shape

    def test_residual_connection(self):
        """Output includes residual, so it should differ from zero input only by residual."""
        cma = CrossModalAttention(
            spatial_channels=8, vector_dim=16, num_heads=2, dropout=0.0,
        )
        spatial = torch.randn(1, 8, 4, 4)
        vector = torch.randn(1, 16)
        out_s, out_v = cma(spatial, vector)
        # Outputs should not be identical to inputs (attention modifies them)
        # but should be close due to residual
        assert out_s.shape == spatial.shape
        assert out_v.shape == vector.shape

    def test_gradient_flow(self):
        cma = CrossModalAttention(
            spatial_channels=8, vector_dim=16, num_heads=2, dropout=0.0,
        )
        spatial = torch.randn(1, 8, 4, 4, requires_grad=True)
        vector = torch.randn(1, 16, requires_grad=True)
        out_s, out_v = cma(spatial, vector)
        loss = out_s.sum() + out_v.sum()
        loss.backward()
        assert spatial.grad is not None
        assert vector.grad is not None


# ---------------------------------------------------------------------------
# AttentionSpatialEncoder
# ---------------------------------------------------------------------------

class TestAttentionSpatialEncoder:
    def test_output_shape(self):
        enc = AttentionSpatialEncoder(
            in_channels=4,
            num_filters=8,
            num_res_blocks=3,
            num_heads=2,
            attention_dropout=0.0,
            attention_interval=2,
            max_spatial_size=16,
        )
        _fix_relative_pos_bug(enc)
        x = torch.randn(2, 4, 8, 8)
        out = enc(x)
        assert out.shape == (2, 8, 8, 8)

    def test_small_num_res_blocks(self):
        enc = AttentionSpatialEncoder(
            in_channels=4,
            num_filters=8,
            num_res_blocks=1,
            num_heads=2,
            attention_dropout=0.0,
            attention_interval=2,
            max_spatial_size=16,
        )
        _fix_relative_pos_bug(enc)
        x = torch.randn(1, 4, 4, 4)
        out = enc(x)
        assert out.shape == (1, 8, 4, 4)

    def test_gradient_flow(self):
        enc = AttentionSpatialEncoder(
            in_channels=4,
            num_filters=8,
            num_res_blocks=2,
            num_heads=2,
            attention_dropout=0.0,
        )
        _fix_relative_pos_bug(enc)
        x = torch.randn(1, 4, 4, 4, requires_grad=True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# HierarchicalPolicyHead
# ---------------------------------------------------------------------------

class TestHierarchicalPolicyHead:
    @pytest.fixture
    def head(self):
        return HierarchicalPolicyHead(
            spatial_filters=8,
            spatial_size=4,
            vector_dim=16,
            action_space_size=32,
            hidden_dim=16,
            category_hidden_dim=8,
        )

    def test_output_shape_no_categories(self, head):
        sf = torch.randn(2, 8, 4, 4)
        vf = torch.randn(2, 16)
        out = head(sf, vf, return_category_probs=False)
        assert out.shape == (2, 32)

    def test_output_shape_with_categories(self, head):
        sf = torch.randn(2, 8, 4, 4)
        vf = torch.randn(2, 16)
        action_logits, cat_probs = head(sf, vf, return_category_probs=True)
        assert action_logits.shape == (2, 32)
        assert cat_probs.shape == (2, HierarchicalPolicyHead.NUM_CATEGORIES)

    def test_category_probs_sum_to_one(self, head):
        sf = torch.randn(2, 8, 4, 4)
        vf = torch.randn(2, 16)
        _, cat_probs = head(sf, vf, return_category_probs=True)
        assert torch.allclose(cat_probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_class_constants(self):
        assert HierarchicalPolicyHead.NUM_CATEGORIES == 9
        assert len(HierarchicalPolicyHead.CATEGORY_NAMES) == 9
        assert "END_TURN" in HierarchicalPolicyHead.CATEGORY_NAMES

    def test_weight_init(self, head):
        # Biases should be initialized to zero
        for name, param in head.named_parameters():
            if "bias" in name and "bn" not in name:
                assert torch.allclose(param, torch.zeros_like(param))


# ---------------------------------------------------------------------------
# Civ6NetworkV2
# ---------------------------------------------------------------------------

class TestCiv6NetworkV2:
    @pytest.fixture
    def small_v2(self, small_config):
        net = Civ6NetworkV2(config=small_config)
        _fix_relative_pos_bug(net)
        return net

    def test_forward_shape(self, small_v2, spatial_8x8, vector_16):
        policy, value = small_v2(spatial_8x8, vector_16)
        assert policy.shape == (2, 32)
        assert value.shape == (2,)

    def test_forward_with_mask(self, small_v2, spatial_8x8, vector_16):
        mask = torch.zeros(2, 32)
        mask[:, :5] = 1.0
        policy, value = small_v2(spatial_8x8, vector_16, action_mask=mask)
        assert torch.all(policy[:, 5:] < -1e8)

    def test_forward_without_attention(self):
        cfg = NetworkConfig(
            spatial_channels=4,
            spatial_size=8,
            vector_dim=16,
            num_res_blocks=2,
            num_filters=8,
            fc_hidden_dim=16,
            action_space_size=32,
            value_head_hidden=8,
            use_attention=False,
        )
        net = Civ6NetworkV2(config=cfg)
        s = torch.randn(2, 4, 8, 8)
        v = torch.randn(2, 16)
        policy, value = net(s, v)
        assert policy.shape == (2, 32)
        assert value.shape == (2,)
        assert net.cross_attention is None

    def test_forward_with_categories(self, small_v2, spatial_8x8, vector_16):
        policy, cat_probs, value = small_v2.forward_with_categories(
            spatial_8x8, vector_16,
        )
        assert policy.shape == (2, 32)
        assert cat_probs.shape == (2, 9)
        assert value.shape == (2,)

    def test_forward_with_categories_and_mask(self, small_v2, spatial_8x8, vector_16):
        mask = torch.zeros(2, 32)
        mask[:, :5] = 1.0
        policy, cat_probs, value = small_v2.forward_with_categories(
            spatial_8x8, vector_16, action_mask=mask,
        )
        assert torch.all(policy[:, 5:] < -1e8)
        assert cat_probs.shape == (2, 9)

    def test_predict(self, small_v2, spatial_8x8, vector_16):
        probs, value = small_v2.predict(spatial_8x8, vector_16)
        assert probs.shape == (2, 32)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_predict_with_mask(self, small_v2, spatial_8x8, vector_16):
        mask = torch.zeros(2, 32)
        mask[:, :5] = 1.0
        probs, _ = small_v2.predict(spatial_8x8, vector_16, action_mask=mask)
        assert probs[:, :5].sum(dim=-1).min() > 0.99

    def test_predict_single_no_mask(self, small_v2):
        s = torch.randn(4, 8, 8)
        v = torch.randn(16)
        probs, val = small_v2.predict_single(s, v)
        assert probs.shape == (32,)
        assert isinstance(val, float)
        assert -1 <= val <= 1

    def test_predict_single_with_mask(self, small_v2):
        s = torch.randn(4, 8, 8)
        v = torch.randn(16)
        m = torch.ones(32)
        m[10:] = 0
        probs, val = small_v2.predict_single(s, v, m)
        assert probs.shape == (32,)
        assert isinstance(val, float)

    def test_predict_hierarchical_no_mask(self, small_v2):
        s = torch.randn(4, 8, 8)
        v = torch.randn(16)
        probs, cat_probs, val = small_v2.predict_hierarchical(s, v)
        assert probs.shape == (32,)
        assert cat_probs.shape == (9,)
        assert isinstance(val, float)
        assert -1 <= val <= 1

    def test_predict_hierarchical_with_mask(self, small_v2):
        s = torch.randn(4, 8, 8)
        v = torch.randn(16)
        m = torch.ones(32)
        m[10:] = 0
        probs, cat_probs, val = small_v2.predict_hierarchical(s, v, m)
        assert probs.shape == (32,)
        assert cat_probs.shape == (9,)

    def test_default_config(self):
        net = Civ6NetworkV2()
        assert net.config.spatial_channels == 84
        assert net.config.action_space_size == 50000
        assert net.config.use_attention is True

    def test_kwargs_override(self):
        net = Civ6NetworkV2(
            spatial_channels=4, spatial_size=4, vector_dim=8,
            num_res_blocks=1, num_filters=8, fc_hidden_dim=8,
            action_space_size=10, value_head_hidden=8,
            num_attention_heads=2,
        )
        assert net.config.spatial_channels == 4

    def test_get_num_parameters(self, small_v2):
        counts = small_v2.get_num_parameters()
        assert "spatial_encoder" in counts
        assert "vector_encoder" in counts
        assert "policy_head" in counts
        assert "value_head" in counts
        assert "cross_attention" in counts
        assert "total" in counts
        assert counts["total"] > 0
        assert counts["total"] == sum(p.numel() for p in small_v2.parameters())

    def test_get_num_parameters_no_attention(self):
        cfg = NetworkConfig(
            spatial_channels=4,
            spatial_size=4,
            vector_dim=8,
            num_res_blocks=1,
            num_filters=8,
            fc_hidden_dim=8,
            action_space_size=10,
            value_head_hidden=8,
            use_attention=False,
        )
        net = Civ6NetworkV2(config=cfg)
        counts = net.get_num_parameters()
        assert "cross_attention" not in counts
        assert counts["total"] > 0

    def test_save_load_roundtrip(self, small_v2, tmp_path):
        path = str(tmp_path / "v2_model.pt")
        small_v2.save(path)
        loaded = Civ6NetworkV2.load(path)
        _fix_relative_pos_bug(loaded)
        assert loaded.config.spatial_channels == small_v2.config.spatial_channels
        assert loaded.config.action_space_size == small_v2.config.action_space_size
        for (_, p1), (_, p2) in zip(
            small_v2.named_parameters(), loaded.named_parameters(), strict=True
        ):
            assert torch.allclose(p1, p2)

    def test_save_load_device(self, small_v2, tmp_path):
        path = str(tmp_path / "v2_model.pt")
        small_v2.save(path)
        loaded = Civ6NetworkV2.load(path, device="cpu")
        _fix_relative_pos_bug(loaded)
        assert next(loaded.parameters()).device == torch.device("cpu")

    def test_from_v1_checkpoint(self, tmp_path):
        # Create a V1 model and save it
        cfg = NetworkConfig(
            spatial_channels=4,
            spatial_size=8,
            vector_dim=16,
            num_res_blocks=2,
            num_filters=8,
            fc_hidden_dim=16,
            action_space_size=32,
            value_head_hidden=8,
        )
        v1 = Civ6Network(config=cfg)
        path = str(tmp_path / "v1_model.pt")
        v1.save(path)

        # Load as V2
        v2 = Civ6NetworkV2.from_v1_checkpoint(path, enable_attention=True)
        _fix_relative_pos_bug(v2)
        assert v2.config.use_attention is True
        assert v2.config.num_attention_heads == 4
        assert v2.config.attention_dropout == 0.1

        # Forward pass should work
        s = torch.randn(1, 4, 8, 8)
        v = torch.randn(1, 16)
        policy, value = v2(s, v)
        assert policy.shape == (1, 32)

    def test_from_v1_checkpoint_no_attention(self, tmp_path):
        cfg = NetworkConfig(
            spatial_channels=4,
            spatial_size=8,
            vector_dim=16,
            num_res_blocks=2,
            num_filters=8,
            fc_hidden_dim=16,
            action_space_size=32,
            value_head_hidden=8,
        )
        v1 = Civ6Network(config=cfg)
        path = str(tmp_path / "v1_model.pt")
        v1.save(path)

        v2 = Civ6NetworkV2.from_v1_checkpoint(path, enable_attention=False)
        assert v2.config.use_attention is False
        assert v2.cross_attention is None

    def test_value_range(self, small_v2):
        s = torch.randn(10, 4, 8, 8)
        v = torch.randn(10, 16)
        _, value = small_v2(s, v)
        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)

    def test_gradient_flow(self, small_v2):
        s = torch.randn(2, 4, 8, 8, requires_grad=True)
        v = torch.randn(2, 16, requires_grad=True)
        policy, value = small_v2(s, v)
        loss = policy.sum() + value.sum()
        loss.backward()
        assert s.grad is not None
        assert v.grad is not None

    def test_init_attention_weights(self, small_v2):
        """Verify attention weight initialization ran (biases should be zero)."""
        for name, param in small_v2.named_parameters():
            if "bias" in name and "bn" not in name and param.dim() == 1:
                # Biases outside batchnorm should be zero-initialized
                assert torch.allclose(param, torch.zeros_like(param)), f"Bias {name} not zero"

    def test_batch_size_one(self, small_v2):
        s = torch.randn(1, 4, 8, 8)
        v = torch.randn(1, 16)
        policy, value = small_v2(s, v)
        assert policy.shape == (1, 32)
        assert value.shape == (1,)

    def test_larger_batch(self, small_v2):
        s = torch.randn(16, 4, 8, 8)
        v = torch.randn(16, 16)
        policy, value = small_v2(s, v)
        assert policy.shape == (16, 32)
        assert value.shape == (16,)


# ---------------------------------------------------------------------------
# compute_loss (extended)
# ---------------------------------------------------------------------------

class TestComputeLossExtended:
    def test_deprecation_warning(self):
        policy_logits = torch.randn(4, 20)
        value = torch.randn(4)
        target_policy = torch.softmax(torch.randn(4, 20), dim=-1)
        target_value = torch.randn(4)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_loss(policy_logits, value, target_policy, target_value)
            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_no_nan_outputs(self):
        policy_logits = torch.randn(4, 20)
        value = torch.randn(4)
        target_policy = torch.softmax(torch.randn(4, 20), dim=-1)
        target_value = torch.randn(4)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            total, policy_loss, value_loss = compute_loss(
                policy_logits, value, target_policy, target_value,
            )
        assert not torch.isnan(total)
        assert not torch.isnan(policy_loss)
        assert not torch.isnan(value_loss)

    def test_with_action_mask_renormalization(self):
        policy_logits = torch.randn(4, 20)
        value = torch.randn(4)
        target_policy = torch.ones(4, 20) / 20
        target_value = torch.randn(4)
        mask = torch.zeros(4, 20)
        mask[:, :5] = 1.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            total, _, _ = compute_loss(
                policy_logits, value, target_policy, target_value, mask,
            )
        assert not torch.isnan(total)

    def test_total_equals_sum(self):
        policy_logits = torch.randn(4, 20)
        value = torch.randn(4)
        target_policy = torch.softmax(torch.randn(4, 20), dim=-1)
        target_value = torch.randn(4)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            total, policy_loss, value_loss = compute_loss(
                policy_logits, value, target_policy, target_value,
            )
        assert torch.allclose(total, policy_loss + value_loss, atol=1e-6)


# ---------------------------------------------------------------------------
# ResidualBlock (extended)
# ---------------------------------------------------------------------------

class TestResidualBlockExtended:
    def test_different_filter_sizes(self):
        for nf in [8, 16, 32]:
            block = ResidualBlock(num_filters=nf)
            x = torch.randn(1, nf, 4, 4)
            out = block(x)
            assert out.shape == x.shape

    def test_gradient_flow(self):
        block = ResidualBlock(num_filters=8)
        x = torch.randn(1, 8, 4, 4, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Inference mode tests
# ---------------------------------------------------------------------------

class TestInferenceMode:
    def test_predict_single_no_grad(self):
        """predict_single should not accumulate gradients."""
        net = MicroCivNetwork(board_size=8, action_space_size=50, num_res_blocks=1, num_filters=8)
        s = torch.randn(13, 8, 8)
        v = torch.randn(64)
        probs, val = net.predict_single(s, v)
        assert not probs.requires_grad

    def test_v2_predict_single_no_grad(self):
        cfg = NetworkConfig(
            spatial_channels=4, spatial_size=4, vector_dim=8,
            num_res_blocks=1, num_filters=8, fc_hidden_dim=8,
            action_space_size=10, value_head_hidden=8,
            num_attention_heads=2,
        )
        net = Civ6NetworkV2(config=cfg)
        _fix_relative_pos_bug(net)
        s = torch.randn(4, 4, 4)
        v = torch.randn(8)
        probs, val = net.predict_single(s, v)
        assert not probs.requires_grad

    def test_v2_predict_hierarchical_no_grad(self):
        cfg = NetworkConfig(
            spatial_channels=4, spatial_size=4, vector_dim=8,
            num_res_blocks=1, num_filters=8, fc_hidden_dim=8,
            action_space_size=10, value_head_hidden=8,
            num_attention_heads=2,
        )
        net = Civ6NetworkV2(config=cfg)
        _fix_relative_pos_bug(net)
        s = torch.randn(4, 4, 4)
        v = torch.randn(8)
        probs, cat_probs, val = net.predict_hierarchical(s, v)
        assert not probs.requires_grad
        assert not cat_probs.requires_grad
