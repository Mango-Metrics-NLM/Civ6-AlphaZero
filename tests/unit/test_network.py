"""Unit tests for neural network."""

import logging
from typing import Any

import pytest
import torch

from alpha_zero.core.civ6_network import (
    AttentionSpatialEncoder,
    Civ6Network,
    Civ6NetworkV2,
    MicroCivNetwork,
    NetworkConfig,
    ResidualBlock,
    SpatialSelfAttention,
    compute_loss,
)


class TestResidualBlock:
    """Test suite for residual block."""

    def test_output_shape(self):
        """Residual block preserves shape."""
        block = ResidualBlock(num_filters=64)
        x = torch.randn(4, 64, 16, 16)
        y = block(x)

        assert y.shape == x.shape

    def test_residual_connection(self):
        """Output includes residual connection."""
        block = ResidualBlock(num_filters=64)

        # Zero out weights to test residual
        with torch.no_grad():
            for param in block.parameters():
                param.zero_()

        x = torch.randn(4, 64, 16, 16)
        y = block(x)

        # Should be close to ReLU(x) due to residual
        assert torch.allclose(y, torch.relu(x), atol=1e-5)


class TestMicroCivNetwork:
    """Test suite for MicroCiv network."""

    @pytest.fixture
    def network(self):
        return MicroCivNetwork(
            board_size=16,
            action_space_size=381,
            num_res_blocks=2,
            num_filters=32,
        )

    def test_forward_shapes(self, network):
        """Forward pass produces correct output shapes."""
        batch_size = 4
        spatial = torch.randn(batch_size, 13, 16, 16)
        vector = torch.randn(batch_size, 64)

        policy_logits, value = network(spatial, vector)

        assert policy_logits.shape == (batch_size, 381)
        assert value.shape == (batch_size,)

    def test_value_range(self, network):
        """Value is in [-1, 1] range (tanh)."""
        spatial = torch.randn(10, 13, 16, 16)
        vector = torch.randn(10, 64)

        _, value = network(spatial, vector)

        assert torch.all(value >= -1)
        assert torch.all(value <= 1)

    def test_action_masking(self, network):
        """Action mask sets invalid actions to low probability."""
        spatial = torch.randn(2, 13, 16, 16)
        vector = torch.randn(2, 64)
        mask = torch.zeros(2, 381)
        mask[:, :10] = 1.0  # Only first 10 actions valid

        policy_logits, _ = network(spatial, vector, mask)

        # Invalid actions should have very negative logits
        assert torch.all(policy_logits[:, 10:] < -1e8)

    def test_predict_returns_probs(self, network):
        """Predict method returns probabilities."""
        spatial = torch.randn(2, 13, 16, 16)
        vector = torch.randn(2, 64)

        probs, _ = network.predict(spatial, vector)

        # Should be valid probability distribution
        assert torch.all(probs >= 0)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    def test_predict_single(self, network):
        """predict_single works without batch dimension."""
        spatial = torch.randn(13, 16, 16)
        vector = torch.randn(64)
        mask = torch.ones(381)
        mask[100:] = 0

        probs, value = network.predict_single(spatial, vector, mask)

        assert probs.shape == (381,)
        assert isinstance(value, float)
        assert -1 <= value <= 1

    def test_gradient_flow(self, network):
        """Gradients flow through network."""
        spatial = torch.randn(2, 13, 16, 16, requires_grad=True)
        vector = torch.randn(2, 64, requires_grad=True)

        policy_logits, value = network(spatial, vector)
        loss = policy_logits.sum() + value.sum()
        loss.backward()

        assert spatial.grad is not None
        assert vector.grad is not None

    def test_save_load(self, network, tmp_path):
        """Network can be saved and loaded."""
        path = tmp_path / "test_model.pt"
        network.save(str(path))

        loaded = MicroCivNetwork.load(str(path))

        # Check weights match
        for (n1, p1), (n2, p2) in zip(
            network.named_parameters(), loaded.named_parameters(), strict=True
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)


class TestCiv6Network:
    """Test suite for full Civ6 network."""

    @pytest.fixture
    def network(self):
        config = NetworkConfig(
            spatial_channels=84,
            spatial_size=44,
            vector_dim=512,
            num_res_blocks=4,
            num_filters=64,
            action_space_size=10000,
        )
        return Civ6Network(config)

    def test_large_action_space(self, network):
        """Network handles large action space."""
        spatial = torch.randn(2, 84, 44, 44)
        vector = torch.randn(2, 512)

        policy_logits, _value = network(spatial, vector)

        assert policy_logits.shape == (2, 10000)

    def test_config_override(self):
        """Kwargs can override config."""
        network = Civ6Network(num_filters=128, spatial_channels=13, spatial_size=16)

        assert network.config.num_filters == 128
        assert network.config.spatial_channels == 13


class TestComputeLoss:
    """Test suite for loss computation."""

    def test_loss_shapes(self):
        """Loss computation produces scalar outputs."""
        policy_logits = torch.randn(8, 100)
        value = torch.randn(8)
        target_policy = torch.softmax(torch.randn(8, 100), dim=-1)
        target_value = torch.randn(8)

        total, policy, value_loss = compute_loss(policy_logits, value, target_policy, target_value)

        assert total.shape == ()
        assert policy.shape == ()
        assert value_loss.shape == ()

    def test_loss_positive(self):
        """Losses are non-negative."""
        policy_logits = torch.randn(8, 100)
        value = torch.randn(8)
        target_policy = torch.softmax(torch.randn(8, 100), dim=-1)
        target_value = torch.randn(8)

        total, _policy, value_loss = compute_loss(policy_logits, value, target_policy, target_value)

        assert total >= 0
        assert value_loss >= 0
        # Policy loss can be negative (cross-entropy)

    def test_loss_with_mask(self):
        """Loss computation respects action mask."""
        policy_logits = torch.randn(4, 50)
        value = torch.randn(4)
        target_policy = torch.zeros(4, 50)
        target_policy[:, :10] = 0.1  # Only first 10 valid
        target_value = torch.randn(4)
        mask = torch.zeros(4, 50)
        mask[:, :10] = 1.0

        total, _, _ = compute_loss(policy_logits, value, target_policy, target_value, mask)

        assert not torch.isnan(total)

    def test_perfect_prediction_low_loss(self):
        """Perfect predictions have low loss."""
        # Create "perfect" prediction
        target_policy = torch.zeros(4, 20)
        target_policy[:, 5] = 1.0  # All mass on action 5

        policy_logits = torch.zeros(4, 20)
        policy_logits[:, 5] = 100.0  # High logit for action 5

        target_value = torch.tensor([0.5, -0.5, 0.0, 1.0])
        value = target_value.clone()

        total, _policy, value_loss = compute_loss(policy_logits, value, target_policy, target_value)

        assert value_loss < 1e-6
        assert total < 1.0  # Should be small


class TestNetworkHardeningM3:
    """M3 regression tests for config immutability, warnings, fp16 masking, save, and V2 attention."""

    @staticmethod
    def _small_v2_config() -> NetworkConfig:
        return NetworkConfig(
            spatial_channels=4,
            spatial_size=6,
            vector_dim=8,
            num_res_blocks=2,
            num_filters=8,
            fc_hidden_dim=16,
            action_space_size=24,
            value_head_hidden=8,
            dropout=0.0,
            num_attention_heads=2,
            use_attention=True,
            attention_dropout=0.0,
        )

    def test_config_not_mutated_by_kwargs(self) -> None:
        cfg = self._small_v2_config()
        v1 = Civ6Network(config=cfg, num_filters=16)
        micro = MicroCivNetwork(config=cfg, fc_hidden_dim=32)
        v2 = Civ6NetworkV2(config=cfg, num_filters=16)

        assert cfg.num_filters == 8
        assert cfg.fc_hidden_dim == 16
        assert v1.config.num_filters == 16
        assert micro.config.fc_hidden_dim == 32
        assert v2.config.num_filters == 16

    def test_unknown_kwargs_warn_and_are_ignored(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="alpha_zero.core.civ6_network"):
            net = Civ6Network(nonexistent_param=999, another_bad_arg=1)

        assert net.config.spatial_channels == 13
        assert any("nonexistent_param" in rec.message for rec in caplog.records)
        assert any("another_bad_arg" in rec.message for rec in caplog.records)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_invalid_action_mask_dtype_safe(self, dtype: torch.dtype) -> None:
        expected_masked = (
            -1e9 if dtype == torch.float32 else float(torch.finfo(torch.float16).min) / 2.0
        )
        cfg = self._small_v2_config()
        v1 = Civ6Network(config=cfg).to(dtype=dtype)
        v2 = Civ6NetworkV2(config=cfg).to(dtype=dtype)
        v1.eval()
        v2.eval()

        spatial = torch.randn(2, 4, 6, 6, dtype=dtype)
        vector = torch.randn(2, 8, dtype=dtype)
        mask = torch.zeros(2, 24, dtype=dtype)
        mask[:, :4] = 1.0

        for logits in (
            v1(spatial, vector, mask)[0],
            v2(spatial, vector, mask)[0],
            v2.forward_with_categories(spatial, vector, mask)[0],
        ):
            assert logits.dtype == dtype
            assert torch.isfinite(logits).all()
            assert torch.allclose(
                logits[:, 4:],
                torch.full_like(logits[:, 4:], expected_masked),
            )
            probs = torch.softmax(logits, dim=-1)
            assert torch.isfinite(probs).all()
            assert torch.all(probs[:, 4:] == 0)

    def test_save_creates_parent_dirs_and_v2_metadata_roundtrip(self, tmp_path) -> None:
        cfg = self._small_v2_config()
        v1 = Civ6Network(config=cfg)
        v1_path = tmp_path / "nested" / "v1" / "model.pt"
        v1.save(str(v1_path), metadata={"max_turns": 15})
        assert v1_path.exists()

        v2 = Civ6NetworkV2(config=cfg)
        v2_path = tmp_path / "nested" / "v2" / "model.pt"
        v2.save(str(v2_path), metadata={"max_turns": 25, "tag": "v2-test"})
        assert v2_path.exists()

        blob = torch.load(str(v2_path), map_location="cpu", weights_only=True)
        assert blob["network_version"] == "v2"
        assert blob["metadata"] == {"max_turns": 25, "tag": "v2-test"}

        with pytest.raises(TypeError):
            v2.save(str(tmp_path / "bad.pt"), metadata={"bad": object()})

    def test_v2_eager_relative_pos_and_save_load_before_and_after_forward(self, tmp_path) -> None:
        cfg = self._small_v2_config()
        v2 = Civ6NetworkV2(config=cfg)

        param_names = dict(v2.named_parameters())
        buffer_names = dict(v2.named_buffers())
        assert any(name.endswith("relative_pos_bias_table") for name in param_names)
        assert any(name.endswith("relative_position_index") for name in buffer_names)

        # Save & load BEFORE any forward()
        pre_path = tmp_path / "pre_forward.pt"
        v2.save(str(pre_path))
        loaded_pre = Civ6NetworkV2.load(str(pre_path))
        for (n1, p1), (n2, p2) in zip(
            v2.named_parameters(), loaded_pre.named_parameters(), strict=True
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)

        # Run forward(), then save & load AFTER forward()
        s = torch.randn(1, 4, 6, 6)
        v = torch.randn(1, 8)
        _ = v2(s, v)
        post_path = tmp_path / "post_forward.pt"
        v2.save(str(post_path))
        loaded_post = Civ6NetworkV2.load(str(post_path))
        out1, _ = v2(s, v)
        out2, _ = loaded_post(s, v)
        assert torch.allclose(out1, out2)

    def test_v2_loads_legacy_preforward_checkpoint_and_rejects_corrupt(
        self, tmp_path, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = self._small_v2_config()
        v2 = Civ6NetworkV2(config=cfg)
        legacy_state = {
            k: val
            for k, val in v2.state_dict().items()
            if not k.endswith(("relative_pos_bias_table", "relative_position_index"))
        }

        target = Civ6NetworkV2(config=cfg)
        with caplog.at_level(logging.WARNING, logger="alpha_zero.core.civ6_network"):
            target.load_state_dict(legacy_state, strict=True)
        assert any("legacy SpatialSelfAttention checkpoint" in r.message for r in caplog.records)

        corrupt_state = {k: val for k, val in legacy_state.items() if not k.endswith("qkv.weight")}
        with pytest.raises(RuntimeError):
            target.load_state_dict(corrupt_state, strict=True)

    def test_spatial_self_attention_dynamic_sizes_without_manual_del(self) -> None:
        attn = SpatialSelfAttention(
            num_channels=8,
            num_heads=2,
            dropout=0.0,
            use_relative_pos=True,
        )
        out4 = attn(torch.randn(1, 8, 4, 4))
        assert out4.shape == (1, 8, 4, 4)
        out3 = attn(torch.randn(1, 8, 3, 3))
        assert out3.shape == (1, 8, 3, 3)

    @pytest.mark.parametrize("hw", [(4, 6), (6, 4), (1, 6), (6, 1)])
    @pytest.mark.parametrize("eager_dst", [True, False])
    def test_nonsquare_attention_checkpoint_preserves_bias_after_forward(
        self,
        hw: tuple[int, int],
        eager_dst: bool,
    ) -> None:
        h, w = hw
        enc_src = AttentionSpatialEncoder(
            in_channels=4,
            num_filters=8,
            num_res_blocks=2,
            num_heads=2,
            attention_dropout=0.0,
            spatial_size=hw,
        )
        with torch.no_grad():
            enc_src.final_attention.relative_pos_bias_table.fill_(2.5)
        sd = {k: v.clone() for k, v in enc_src.state_dict().items()}

        enc_dst = AttentionSpatialEncoder(
            in_channels=4,
            num_filters=8,
            num_res_blocks=2,
            num_heads=2,
            attention_dropout=0.0,
            spatial_size=hw if eager_dst else None,
        )
        enc_dst.load_state_dict(sd, strict=True)
        assert enc_dst.final_attention._spatial_hw in (
            (hw,) if min(hw) > 1 or eager_dst else ((1, max(hw)), (max(hw), 1))
        )
        assert torch.allclose(
            enc_dst.final_attention.relative_pos_bias_table,
            sd["final_attention.relative_pos_bias_table"],
        )

        enc_dst.eval()
        enc_src.eval()
        x = torch.randn(1, 4, h, w)
        out_src = enc_src(x)
        out_dst = enc_dst(x)
        assert enc_dst.final_attention._spatial_hw == hw
        assert torch.allclose(
            enc_dst.final_attention.relative_pos_bias_table,
            sd["final_attention.relative_pos_bias_table"],
        )
        assert torch.allclose(out_src, out_dst)

    @pytest.mark.parametrize("dst_spatial_size", [4, None])
    def test_spatial_self_attention_rejects_mismatched_num_heads_and_invalid_shapes(
        self, dst_spatial_size: int | None
    ) -> None:
        src = SpatialSelfAttention(8, num_heads=4, spatial_size=4)
        dst = SpatialSelfAttention(8, num_heads=2, spatial_size=dst_spatial_size)
        with pytest.raises(RuntimeError):
            dst.load_state_dict(src.state_dict(), strict=True)

        bad_rank_sd = {k: v.clone() for k, v in dst.state_dict().items()}
        src_2h = SpatialSelfAttention(8, num_heads=2, spatial_size=4)
        bad_rank_sd["relative_pos_bias_table"] = src_2h.relative_pos_bias_table.detach().reshape(-1)
        bad_rank_sd["relative_position_index"] = src_2h.relative_position_index.detach().clone()
        with pytest.raises(RuntimeError):
            dst.load_state_dict(bad_rank_sd, strict=True)

        bad_idx_sd = {k: v.clone() for k, v in src_2h.state_dict().items()}
        bad_idx_sd["relative_position_index"] = torch.full_like(
            bad_idx_sd["relative_position_index"], 999
        )
        with pytest.raises(RuntimeError):
            dst.load_state_dict(bad_idx_sd, strict=True)

    @pytest.mark.parametrize("strict", [True, False])
    @pytest.mark.parametrize(
        "bad_index",
        [
            torch.full((256,), 999.0, dtype=torch.float32),
            torch.ones((256,), dtype=torch.bool),
            "not_a_tensor",
        ],
    )
    def test_eager_spatial_attention_rejects_partial_invalid_index_without_mutation(
        self, strict: bool, bad_index: Any
    ) -> None:
        dst = SpatialSelfAttention(8, num_heads=2, spatial_size=4)
        orig_table = dst.relative_pos_bias_table.detach().clone()
        orig_index = dst.relative_position_index.detach().clone()

        partial_sd: dict[str, Any] = {
            k: v.clone() for k, v in dst.state_dict().items() if k != "relative_pos_bias_table"
        }
        partial_sd["relative_position_index"] = bad_index

        with pytest.raises(RuntimeError):
            dst.load_state_dict(partial_sd, strict=strict)

        assert dst._spatial_hw == (4, 4)
        assert torch.equal(dst.relative_pos_bias_table, orig_table)
        assert torch.equal(dst.relative_position_index, orig_index)

    @pytest.mark.parametrize("strict", [True, False])
    @pytest.mark.parametrize(
        "bad_table",
        [
            "not_a_tensor",
            None,
            torch.ones((49,), dtype=torch.float32),
            torch.ones((49, 4), dtype=torch.float32),
        ],
    )
    def test_eager_spatial_attention_rejects_partial_invalid_table_without_mutation(
        self, strict: bool, bad_table: Any
    ) -> None:
        dst = SpatialSelfAttention(8, num_heads=2, spatial_size=4)
        orig_table = dst.relative_pos_bias_table.detach().clone()
        orig_index = dst.relative_position_index.detach().clone()

        partial_sd: dict[str, Any] = {
            k: v.clone() for k, v in dst.state_dict().items() if k != "relative_position_index"
        }
        partial_sd["relative_pos_bias_table"] = bad_table

        with pytest.raises(RuntimeError):
            dst.load_state_dict(partial_sd, strict=strict)

        assert dst._spatial_hw == (4, 4)
        assert torch.equal(dst.relative_pos_bias_table, orig_table)
        assert torch.equal(dst.relative_position_index, orig_index)

    @pytest.mark.parametrize("strict", [True, False])
    def test_eager_spatial_attention_rejects_matching_shape_float_index_with_table_no_mutation(
        self, strict: bool
    ) -> None:
        dst = SpatialSelfAttention(8, num_heads=2, spatial_size=4)
        orig_table = dst.relative_pos_bias_table.detach().clone()
        orig_index = dst.relative_position_index.detach().clone()

        bad_sd: dict[str, Any] = {k: v.clone() for k, v in dst.state_dict().items()}
        bad_sd["relative_pos_bias_table"] = torch.full_like(orig_table, 3.25)
        bad_sd["relative_position_index"] = torch.full((256,), 999.0, dtype=torch.float32)

        with pytest.raises(RuntimeError):
            dst.load_state_dict(bad_sd, strict=strict)

        assert dst._spatial_hw == (4, 4)
        assert torch.equal(dst.relative_pos_bias_table, orig_table)
        assert torch.equal(dst.relative_position_index, orig_index)
