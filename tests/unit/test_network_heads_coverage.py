"""Coverage tests for alpha_zero.core.network_heads module."""

import pytest
import torch

from alpha_zero.core.network_heads import (
    ActivationType,
    FeatureCombiner,
    HeadFactory,
    MLPHead,
    MLPHeadConfig,
    SpatialPreprocessor,
    SpatialPreprocessorConfig,
    _CNNHead,
    get_activation,
    get_activation_module,
)


class TestActivationType:
    def test_enum_values(self):
        assert ActivationType.RELU == "relu"
        assert ActivationType.GELU == "gelu"
        assert ActivationType.TANH == "tanh"
        assert ActivationType.SIGMOID == "sigmoid"
        assert ActivationType.NONE == "none"


class TestGetActivation:
    @pytest.mark.parametrize("name", ["relu", "gelu", "tanh", "sigmoid", "none"])
    def test_all_activations_from_string(self, name):
        fn = get_activation(name)
        t = torch.randn(4)
        result = fn(t)
        assert result.shape == t.shape

    @pytest.mark.parametrize("act", list(ActivationType))
    def test_all_activations_from_enum(self, act):
        fn = get_activation(act)
        t = torch.randn(4)
        result = fn(t)
        assert result.shape == t.shape

    def test_unknown_activation_string_raises(self):
        with pytest.raises(ValueError):
            get_activation("nonexistent_activation")

    def test_relu_correctness(self):
        fn = get_activation("relu")
        t = torch.tensor([-1.0, 0.0, 1.0])
        result = fn(t)
        assert result[0].item() == 0.0
        assert result[2].item() == 1.0

    def test_tanh_range(self):
        fn = get_activation("tanh")
        t = torch.tensor([100.0, -100.0])
        result = fn(t)
        assert -1.0 <= result[0].item() <= 1.0

    def test_none_is_identity(self):
        fn = get_activation("none")
        t = torch.tensor([3.14])
        assert fn(t).item() == pytest.approx(3.14)


class TestGetActivationModule:
    @pytest.mark.parametrize("name", ["relu", "gelu", "tanh", "sigmoid", "none"])
    def test_all_modules_from_string(self, name):
        mod = get_activation_module(name)
        assert isinstance(mod, torch.nn.Module)
        t = torch.randn(4)
        result = mod(t)
        assert result.shape == t.shape

    @pytest.mark.parametrize("act", list(ActivationType))
    def test_all_modules_from_enum(self, act):
        mod = get_activation_module(act)
        assert isinstance(mod, torch.nn.Module)


class TestMLPHeadConfig:
    def test_defaults(self):
        config = MLPHeadConfig(input_dim=64, output_dim=10)
        assert config.hidden_dims == [256]
        assert config.activation == "relu"
        assert config.output_activation == "none"
        assert config.dropout == 0.0
        assert config.use_layer_norm is False
        assert config.use_batch_norm is False

    def test_invalid_input_dim(self):
        with pytest.raises(ValueError, match="input_dim must be positive"):
            MLPHeadConfig(input_dim=0, output_dim=10)

    def test_invalid_output_dim(self):
        with pytest.raises(ValueError, match="output_dim must be positive"):
            MLPHeadConfig(input_dim=64, output_dim=-1)

    def test_invalid_hidden_dims(self):
        with pytest.raises(ValueError, match="hidden_dims must be positive"):
            MLPHeadConfig(input_dim=64, output_dim=10, hidden_dims=[128, 0])

    def test_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout must be in"):
            MLPHeadConfig(input_dim=64, output_dim=10, dropout=1.0)

    def test_both_norms_raises(self):
        with pytest.raises(ValueError, match="Cannot use both"):
            MLPHeadConfig(input_dim=64, output_dim=10, use_layer_norm=True, use_batch_norm=True)


class TestMLPHead:
    def test_basic_forward(self):
        config = MLPHeadConfig(input_dim=64, output_dim=10, hidden_dims=[32])
        head = MLPHead(config)
        x = torch.randn(8, 64)
        out = head(x)
        assert out.shape == (8, 10)

    def test_multiple_hidden_layers(self):
        config = MLPHeadConfig(input_dim=128, output_dim=5, hidden_dims=[64, 32, 16])
        head = MLPHead(config)
        x = torch.randn(4, 128)
        out = head(x)
        assert out.shape == (4, 5)

    def test_with_dropout(self):
        config = MLPHeadConfig(input_dim=64, output_dim=10, hidden_dims=[32], dropout=0.5)
        head = MLPHead(config)
        head.train()
        x = torch.randn(8, 64)
        out = head(x)
        assert out.shape == (8, 10)

    def test_with_layer_norm(self):
        config = MLPHeadConfig(input_dim=64, output_dim=10, hidden_dims=[32], use_layer_norm=True)
        head = MLPHead(config)
        x = torch.randn(8, 64)
        out = head(x)
        assert out.shape == (8, 10)

    def test_with_batch_norm(self):
        config = MLPHeadConfig(input_dim=64, output_dim=10, hidden_dims=[32], use_batch_norm=True)
        head = MLPHead(config)
        head.train()
        x = torch.randn(8, 64)
        out = head(x)
        assert out.shape == (8, 10)

    def test_with_tanh_output(self):
        config = MLPHeadConfig(
            input_dim=64, output_dim=1, hidden_dims=[32], output_activation="tanh"
        )
        head = MLPHead(config)
        x = torch.randn(8, 64)
        out = head(x)
        assert out.shape == (8, 1)
        assert (out >= -1.0).all() and (out <= 1.0).all()

    def test_gelu_activation(self):
        config = MLPHeadConfig(input_dim=64, output_dim=10, hidden_dims=[32], activation="gelu")
        head = MLPHead(config)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 10)

    def test_stores_config(self):
        config = MLPHeadConfig(input_dim=64, output_dim=10)
        head = MLPHead(config)
        assert head.config is config


class TestSpatialPreprocessorConfig:
    def test_defaults(self):
        config = SpatialPreprocessorConfig(in_channels=64, out_channels=2, spatial_size=16)
        assert config.use_batch_norm is True

    def test_custom(self):
        config = SpatialPreprocessorConfig(
            in_channels=128, out_channels=4, spatial_size=8, use_batch_norm=False
        )
        assert config.in_channels == 128
        assert config.use_batch_norm is False


class TestSpatialPreprocessor:
    def test_forward_with_bn(self):
        config = SpatialPreprocessorConfig(in_channels=64, out_channels=2, spatial_size=16)
        prep = SpatialPreprocessor(config)
        prep.train()
        x = torch.randn(8, 64, 16, 16)
        out = prep(x)
        assert out.shape == (8, 2 * 16 * 16)

    def test_forward_without_bn(self):
        config = SpatialPreprocessorConfig(
            in_channels=64, out_channels=2, spatial_size=16, use_batch_norm=False
        )
        prep = SpatialPreprocessor(config)
        x = torch.randn(4, 64, 16, 16)
        out = prep(x)
        assert out.shape == (4, 2 * 16 * 16)

    def test_output_dim_attribute(self):
        config = SpatialPreprocessorConfig(in_channels=32, out_channels=4, spatial_size=8)
        prep = SpatialPreprocessor(config)
        assert prep.output_dim == 4 * 8 * 8


class TestFeatureCombiner:
    def test_without_projection(self):
        combiner = FeatureCombiner(spatial_dim=512, vector_dim=64)
        spatial = torch.randn(4, 512)
        vector = torch.randn(4, 64)
        out = combiner(spatial, vector)
        assert out.shape == (4, 576)
        assert combiner.output_dim == 576
        assert combiner.projection is None

    def test_with_projection(self):
        combiner = FeatureCombiner(spatial_dim=512, vector_dim=64, projection_dim=128)
        spatial = torch.randn(4, 512)
        vector = torch.randn(4, 64)
        out = combiner(spatial, vector)
        assert out.shape == (4, 128)
        assert combiner.output_dim == 128
        assert combiner.projection is not None


class TestCNNHead:
    def test_forward(self):
        sp_config = SpatialPreprocessorConfig(in_channels=64, out_channels=2, spatial_size=8)
        sp = SpatialPreprocessor(sp_config)
        combiner = FeatureCombiner(spatial_dim=sp.output_dim, vector_dim=32)
        mlp_config = MLPHeadConfig(input_dim=combiner.output_dim, output_dim=10, hidden_dims=[64])
        mlp = MLPHead(mlp_config)
        head = _CNNHead(sp, combiner, mlp)
        spatial = torch.randn(4, 64, 8, 8)
        vector = torch.randn(4, 32)
        out = head(spatial, vector)
        assert out.shape == (4, 10)


class TestHeadFactory:
    def test_create_cnn_policy_head(self):
        head = HeadFactory.create_cnn_policy_head(
            spatial_filters=64, spatial_size=16, vector_dim=32, action_space_size=100
        )
        spatial = torch.randn(4, 64, 16, 16)
        vector = torch.randn(4, 32)
        out = head(spatial, vector)
        assert out.shape == (4, 100)

    def test_create_cnn_value_head(self):
        head = HeadFactory.create_cnn_value_head(spatial_filters=64, spatial_size=16, vector_dim=32)
        spatial = torch.randn(4, 64, 16, 16)
        vector = torch.randn(4, 32)
        out = head(spatial, vector)
        assert out.shape == (4, 1)

    def test_create_transformer_policy_head(self):
        head = HeadFactory.create_transformer_policy_head(embed_dim=256, action_space_size=100)
        x = torch.randn(4, 256)
        out = head(x)
        assert out.shape == (4, 100)

    def test_create_transformer_value_head(self):
        head = HeadFactory.create_transformer_value_head(embed_dim=256)
        x = torch.randn(4, 256)
        out = head(x)
        assert out.shape == (4, 1)

    def test_cnn_policy_custom_hidden_dim(self):
        head = HeadFactory.create_cnn_policy_head(
            spatial_filters=32,
            spatial_size=8,
            vector_dim=16,
            action_space_size=50,
            hidden_dim=128,
            spatial_reduction_channels=4,
        )
        spatial = torch.randn(2, 32, 8, 8)
        vector = torch.randn(2, 16)
        out = head(spatial, vector)
        assert out.shape == (2, 50)

    def test_transformer_policy_with_dropout(self):
        head = HeadFactory.create_transformer_policy_head(
            embed_dim=128, action_space_size=50, hidden_dim=64, dropout=0.2
        )
        head.train()
        x = torch.randn(4, 128)
        out = head(x)
        assert out.shape == (4, 50)

    def test_transformer_value_with_dropout(self):
        head = HeadFactory.create_transformer_value_head(embed_dim=128, hidden_dim=64, dropout=0.2)
        x = torch.randn(4, 128)
        out = head(x)
        assert out.shape == (4, 1)
