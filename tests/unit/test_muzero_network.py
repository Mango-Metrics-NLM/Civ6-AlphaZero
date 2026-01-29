"""Unit tests for MuZero network architecture."""

import pytest
import torch

from alpha_zero.core.muzero_network import (
    ActionEmbedding,
    DynamicsNetwork,
    MuZeroConfig,
    MuZeroNetwork,
    PredictionHead,
    ProjectionHead,
    RewardHead,
    ValuePrefixHead,
    categorical_to_scalar,
    compute_consistency_loss,
    compute_muzero_loss,
    scalar_to_categorical,
)


class TestMuZeroConfig:
    """Test suite for MuZero configuration."""

    def test_default_values(self):
        """Default config has reasonable values."""
        config = MuZeroConfig()

        assert config.spatial_channels == 13
        assert config.spatial_size == 16
        assert config.hidden_state_channels == 64
        assert config.dynamics_num_blocks == 4
        assert config.use_consistency_loss is True
        assert config.use_value_prefix is True

    def test_custom_values(self):
        """Config accepts custom values."""
        config = MuZeroConfig(
            hidden_state_channels=128,
            dynamics_num_blocks=8,
            use_categorical_reward=False,
        )

        assert config.hidden_state_channels == 128
        assert config.dynamics_num_blocks == 8
        assert config.use_categorical_reward is False


class TestActionEmbedding:
    """Test suite for action embedding."""

    def test_output_shape(self):
        """Output has correct shape."""
        embed = ActionEmbedding(
            action_space_size=100,
            embedding_dim=32,
            spatial_size=16,
        )

        actions = torch.tensor([0, 5, 10])
        output = embed(actions)

        assert output.shape == (3, 32, 16, 16)

    def test_different_actions_different_embeddings(self):
        """Different actions produce different embeddings."""
        embed = ActionEmbedding(
            action_space_size=100,
            embedding_dim=32,
            spatial_size=8,
        )

        action1 = torch.tensor([0])
        action2 = torch.tensor([1])

        out1 = embed(action1)
        out2 = embed(action2)

        assert not torch.allclose(out1, out2)


class TestDynamicsNetwork:
    """Test suite for dynamics network (world model)."""

    @pytest.fixture
    def dynamics(self):
        return DynamicsNetwork(
            hidden_channels=64,
            action_space_size=100,
            action_embedding_dim=32,
            spatial_size=16,
            num_blocks=2,
            reward_support_size=601,
            use_categorical_reward=True,
        )

    @pytest.fixture
    def dynamics_scalar(self):
        return DynamicsNetwork(
            hidden_channels=64,
            action_space_size=100,
            action_embedding_dim=32,
            spatial_size=16,
            num_blocks=2,
            use_categorical_reward=False,
        )

    def test_output_shapes_categorical(self, dynamics):
        """Categorical reward outputs have correct shapes."""
        hidden = torch.randn(4, 64, 16, 16)
        actions = torch.randint(0, 100, (4,))

        output = dynamics(hidden, actions)

        assert output.next_hidden_state.shape == (4, 64, 16, 16)
        assert output.reward.shape == (4, 601)

    def test_output_shapes_scalar(self, dynamics_scalar):
        """Scalar reward outputs have correct shapes."""
        hidden = torch.randn(4, 64, 16, 16)
        actions = torch.randint(0, 100, (4,))

        output = dynamics_scalar(hidden, actions)

        assert output.next_hidden_state.shape == (4, 64, 16, 16)
        assert output.reward.shape == (4,)

    def test_different_actions_different_outputs(self, dynamics):
        """Different actions produce different next states."""
        hidden = torch.randn(1, 64, 16, 16)
        action1 = torch.tensor([0])
        action2 = torch.tensor([50])

        out1 = dynamics(hidden, action1)
        out2 = dynamics(hidden, action2)

        assert not torch.allclose(out1.next_hidden_state, out2.next_hidden_state)


class TestRewardHead:
    """Test suite for reward prediction head."""

    def test_categorical_output(self):
        """Categorical reward head outputs correct shape."""
        head = RewardHead(
            hidden_channels=64,
            spatial_size=16,
            support_size=601,
            use_categorical=True,
        )

        hidden = torch.randn(4, 64, 16, 16)
        reward = head(hidden)

        assert reward.shape == (4, 601)

    def test_scalar_output(self):
        """Scalar reward head outputs correct shape."""
        head = RewardHead(
            hidden_channels=64,
            spatial_size=16,
            use_categorical=False,
        )

        hidden = torch.randn(4, 64, 16, 16)
        reward = head(hidden)

        assert reward.shape == (4,)


class TestProjectionHead:
    """Test suite for projection head."""

    def test_output_shape(self):
        """Projection head outputs correct shape."""
        head = ProjectionHead(
            hidden_channels=64,
            spatial_size=16,
            projection_dim=256,
        )

        hidden = torch.randn(4, 64, 16, 16)
        projection = head(hidden)

        assert projection.shape == (4, 256)


class TestPredictionHead:
    """Test suite for prediction head (SimSiam-style)."""

    def test_output_shape(self):
        """Prediction head outputs correct shape."""
        head = PredictionHead(projection_dim=256, hidden_dim=128)

        projection = torch.randn(4, 256)
        prediction = head(projection)

        assert prediction.shape == (4, 256)


class TestValuePrefixHead:
    """Test suite for value prefix prediction."""

    def test_categorical_output(self):
        """Categorical value prefix outputs correct shape."""
        head = ValuePrefixHead(
            hidden_channels=64,
            spatial_size=16,
            horizon=5,
            support_size=601,
            use_categorical=True,
        )

        hidden = torch.randn(4, 64, 16, 16)
        prefix = head(hidden)

        assert prefix.shape == (4, 601)

    def test_scalar_output(self):
        """Scalar value prefix outputs correct shape."""
        head = ValuePrefixHead(
            hidden_channels=64,
            spatial_size=16,
            horizon=5,
            use_categorical=False,
        )

        hidden = torch.randn(4, 64, 16, 16)
        prefix = head(hidden)

        assert prefix.shape == (4,)


class TestMuZeroNetwork:
    """Test suite for full MuZero network."""

    @pytest.fixture
    def config(self):
        return MuZeroConfig(
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            num_res_blocks=2,
            hidden_state_channels=32,
            fc_hidden_dim=64,
            action_space_size=100,
            dynamics_num_blocks=2,
            use_consistency_loss=True,
            use_value_prefix=True,
        )

    @pytest.fixture
    def network(self, config):
        return MuZeroNetwork(config)

    def test_representation(self, network):
        """Representation function outputs correct shape."""
        spatial = torch.randn(4, 13, 8, 8)
        vector = torch.randn(4, 64)

        hidden = network.representation(spatial, vector)

        assert hidden.shape == (4, 32, 8, 8)

    def test_dynamics_function(self, network):
        """Dynamics function outputs correct shapes."""
        hidden = torch.randn(4, 32, 8, 8)
        actions = torch.randint(0, 100, (4,))

        output = network.dynamics_function(hidden, actions)

        assert output.next_hidden_state.shape == (4, 32, 8, 8)
        assert output.reward.shape[0] == 4  # Categorical has shape (4, 601)

    def test_prediction_function(self, network):
        """Prediction function outputs correct shapes."""
        hidden = torch.randn(4, 32, 8, 8)
        mask = torch.ones(4, 100)

        policy, value = network.prediction_function(hidden, mask)

        assert policy.shape == (4, 100)
        assert value.shape == (4,)

    def test_initial_inference(self, network):
        """Initial inference outputs correct shapes."""
        spatial = torch.randn(4, 13, 8, 8)
        vector = torch.randn(4, 64)
        mask = torch.ones(4, 100)

        output = network.initial_inference(spatial, vector, mask)

        assert output.hidden_state.shape == (4, 32, 8, 8)
        assert output.policy_logits.shape == (4, 100)
        assert output.value.shape == (4,)

    def test_recurrent_inference(self, network):
        """Recurrent inference outputs correct shapes."""
        hidden = torch.randn(4, 32, 8, 8)
        actions = torch.randint(0, 100, (4,))
        mask = torch.ones(4, 100)

        next_hidden, reward, policy, value = network.recurrent_inference(
            hidden, actions, mask
        )

        assert next_hidden.shape == (4, 32, 8, 8)
        assert policy.shape == (4, 100)
        assert value.shape == (4,)

    def test_forward_compatibility(self, network):
        """Forward method is AlphaZero-compatible."""
        spatial = torch.randn(4, 13, 8, 8)
        vector = torch.randn(4, 64)
        mask = torch.ones(4, 100)

        policy_logits, value = network(spatial, vector, mask)

        assert policy_logits.shape == (4, 100)
        assert value.shape == (4,)

    def test_predict_compatibility(self, network):
        """Predict method is AlphaZero-compatible."""
        spatial = torch.randn(4, 13, 8, 8)
        vector = torch.randn(4, 64)
        mask = torch.ones(4, 100)

        policy_probs, value = network.predict(spatial, vector, mask)

        assert policy_probs.shape == (4, 100)
        assert torch.allclose(policy_probs.sum(dim=-1), torch.ones(4), atol=1e-5)
        assert value.shape == (4,)

    def test_action_mask_applied(self, network):
        """Action mask is correctly applied."""
        spatial = torch.randn(1, 13, 8, 8)
        vector = torch.randn(1, 64)
        mask = torch.zeros(1, 100)
        mask[0, 0] = 1  # Only action 0 valid

        policy_probs, _ = network.predict(spatial, vector, mask)

        # Action 0 should have nearly all probability
        assert policy_probs[0, 0] > 0.99

    def test_project(self, network):
        """Projection works when consistency loss enabled."""
        hidden = torch.randn(4, 32, 8, 8)
        projection = network.project(hidden)

        assert projection.shape == (4, 256)

    def test_predict_projection(self, network):
        """Prediction head works when consistency loss enabled."""
        projection = torch.randn(4, 256)
        prediction = network.predict_projection(projection)

        assert prediction.shape == (4, 256)

    def test_predict_value_prefix(self, network):
        """Value prefix prediction works when enabled."""
        hidden = torch.randn(4, 32, 8, 8)
        prefix = network.predict_value_prefix(hidden)

        assert prefix.shape[0] == 4

    def test_get_num_parameters(self, network):
        """Parameter counting works."""
        counts = network.get_num_parameters()

        assert "total" in counts
        assert "spatial_encoder" in counts
        assert "dynamics" in counts
        assert counts["total"] > 0


class TestMuZeroNetworkWithoutExtensions:
    """Test MuZero network without EfficientZero extensions."""

    @pytest.fixture
    def network(self):
        config = MuZeroConfig(
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            num_res_blocks=2,
            hidden_state_channels=32,
            fc_hidden_dim=64,
            action_space_size=100,
            use_consistency_loss=False,
            use_value_prefix=False,
        )
        return MuZeroNetwork(config)

    def test_no_projection_head(self, network):
        """No projection head when consistency loss disabled."""
        assert network.projection_head is None

    def test_no_value_prefix_head(self, network):
        """No value prefix head when disabled."""
        assert network.value_prefix_head is None

    def test_project_raises(self, network):
        """Project raises when consistency loss disabled."""
        hidden = torch.randn(4, 32, 8, 8)
        with pytest.raises(ValueError):
            network.project(hidden)


class TestScalarCategoricalConversion:
    """Test suite for scalar-categorical conversion utilities."""

    def test_scalar_to_categorical_roundtrip(self):
        """Scalar -> categorical -> scalar preserves values."""
        scalars = torch.tensor([0.0, 10.0, -50.0, 100.0, -100.0])
        categorical = scalar_to_categorical(scalars, support_size=601)
        recovered = categorical_to_scalar(categorical, support_size=601)

        assert torch.allclose(scalars, recovered, atol=2.0)

    def test_categorical_sums_to_one(self):
        """Categorical distribution sums to 1."""
        scalars = torch.tensor([5.0, -5.0, 0.0])
        categorical = scalar_to_categorical(scalars, support_size=601)

        assert torch.allclose(categorical.sum(dim=-1), torch.ones(3), atol=1e-5)

    def test_extreme_values_clamped(self):
        """Extreme values are clamped to support range."""
        scalars = torch.tensor([1000.0, -1000.0])
        categorical = scalar_to_categorical(scalars, support_size=601, min_value=-300, max_value=300)

        # Should not crash, values clamped
        assert categorical.shape == (2, 601)


class TestMuZeroLoss:
    """Test suite for MuZero loss computation."""

    def test_loss_computation_categorical(self):
        """Loss computation works with categorical predictions."""
        policy_logits = torch.randn(4, 100, requires_grad=True)
        value = torch.randn(4, 601, requires_grad=True)
        reward = torch.randn(4, 601, requires_grad=True)

        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)
        target_reward = torch.randn(4)

        losses = compute_muzero_loss(
            policy_logits=policy_logits,
            value=value,
            reward=reward,
            target_policy=target_policy,
            target_value=target_value,
            target_reward=target_reward,
            use_categorical=True,
        )

        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "reward_loss" in losses
        # All losses should be scalar tensors
        assert all(loss.dim() == 0 for loss in losses.values())

    def test_loss_without_reward(self):
        """Loss computation works without reward (initial step)."""
        policy_logits = torch.randn(4, 100)
        value = torch.randn(4, 601)

        target_policy = torch.softmax(torch.randn(4, 100), dim=-1)
        target_value = torch.randn(4)

        losses = compute_muzero_loss(
            policy_logits=policy_logits,
            value=value,
            reward=None,
            target_policy=target_policy,
            target_value=target_value,
            use_categorical=True,
        )

        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "reward_loss" not in losses


class TestConsistencyLoss:
    """Test suite for EfficientZero consistency loss."""

    def test_consistency_loss_identical_inputs(self):
        """Consistency loss is minimal for identical inputs."""
        projection = torch.randn(4, 256)
        target = projection.clone()

        loss = compute_consistency_loss(projection, target)

        # Should be close to -1 (negative cosine similarity = -1 for identical)
        assert loss < -0.99

    def test_consistency_loss_different_inputs(self):
        """Consistency loss is larger for different inputs."""
        projection = torch.randn(4, 256)
        target = torch.randn(4, 256)

        loss = compute_consistency_loss(projection, target)

        # Should be larger than for identical inputs
        assert loss > -1.0


class TestMuZeroNetworkSaveLoad:
    """Test saving and loading MuZero networks."""

    def test_save_load_roundtrip(self, tmp_path):
        """Network can be saved and loaded."""
        config = MuZeroConfig(
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            num_res_blocks=2,
            hidden_state_channels=32,
            action_space_size=100,
        )
        network = MuZeroNetwork(config)

        # Save
        path = tmp_path / "muzero_test.pt"
        network.save(str(path))

        # Load
        loaded = MuZeroNetwork.load(str(path))

        # Check config preserved
        assert loaded.config.spatial_channels == config.spatial_channels
        assert loaded.config.hidden_state_channels == config.hidden_state_channels

        # Check weights preserved
        spatial = torch.randn(1, 13, 8, 8)
        vector = torch.randn(1, 64)

        network.eval()
        loaded.eval()

        with torch.no_grad():
            out1, val1 = network(spatial, vector)
            out2, val2 = loaded(spatial, vector)

        assert torch.allclose(out1, out2)
        assert torch.allclose(val1, val2)
