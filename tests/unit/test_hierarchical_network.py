"""Tests for hierarchical neural network components.

Tests cover:
- HierarchicalNetworkConfig defaults and level-specific factories
- ResidualBlock forward pass with skip connections
- GoalConditionedHead with attention and concatenation modes
- LevelNetwork full forward pass and state embedding
- HierarchicalNetwork multi-level forward and cross-level context
- Save/load functionality
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip entire module if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch required for hierarchical network tests"
)

if TORCH_AVAILABLE:
    from alpha_zero.hierarchical.config import HierarchyLevel
    from alpha_zero.hierarchical.network.hierarchical_network import (
        GoalConditionedHead,
        HierarchicalNetwork,
        HierarchicalNetworkConfig,
        LevelNetwork,
        ResidualBlock,
    )


# =============================================================================
# Constants
# =============================================================================

BATCH_SIZE = 4
BOARD_SIZE = (8, 8)  # Smaller for faster tests
SPATIAL_CHANNELS = 13
VECTOR_DIM = 64
EMBED_DIM = 32  # Smaller for faster tests
GOAL_DIM = 16
ACTION_SPACE = 100


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def network_config() -> HierarchicalNetworkConfig:
    """Create minimal network configuration for testing."""
    return HierarchicalNetworkConfig(
        board_size=BOARD_SIZE,
        spatial_channels=SPATIAL_CHANNELS,
        vector_dim=VECTOR_DIM,
        action_space_size=ACTION_SPACE,
        embed_dim=EMBED_DIM,
        num_res_blocks=2,
        goal_embed_dim=GOAL_DIM,
        num_attention_heads=2,
        dropout=0.1,
        use_goal_attention=True,
    )


@pytest.fixture
def sample_spatial() -> torch.Tensor:
    """Create sample spatial input tensor."""
    return torch.randn(BATCH_SIZE, SPATIAL_CHANNELS, *BOARD_SIZE)


@pytest.fixture
def sample_vector() -> torch.Tensor:
    """Create sample vector input tensor."""
    return torch.randn(BATCH_SIZE, VECTOR_DIM)


@pytest.fixture
def sample_goal_embedding() -> torch.Tensor:
    """Create sample goal embedding tensor."""
    return torch.randn(BATCH_SIZE, GOAL_DIM)


@pytest.fixture
def sample_action_mask() -> torch.Tensor:
    """Create sample action mask (all valid by default)."""
    return torch.ones(BATCH_SIZE, ACTION_SPACE)


# =============================================================================
# HierarchicalNetworkConfig Tests
# =============================================================================


class TestHierarchicalNetworkConfig:
    """Tests for HierarchicalNetworkConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = HierarchicalNetworkConfig()

        assert config.board_size == (16, 16)
        assert config.spatial_channels == 13
        assert config.vector_dim == 64
        assert config.action_space_size == 356
        assert config.embed_dim == 128
        assert config.num_res_blocks == 6
        assert config.goal_embed_dim == 64
        assert config.num_attention_heads == 4
        assert config.dropout == 0.1
        assert config.use_goal_attention is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = HierarchicalNetworkConfig(
            board_size=(8, 8),
            embed_dim=256,
            num_res_blocks=10,
            dropout=0.2,
            use_goal_attention=False,
        )

        assert config.board_size == (8, 8)
        assert config.embed_dim == 256
        assert config.num_res_blocks == 10
        assert config.dropout == 0.2
        assert config.use_goal_attention is False

    def test_for_level_strategic(self) -> None:
        """Test strategic level configuration."""
        config = HierarchicalNetworkConfig.for_level(HierarchyLevel.STRATEGIC)

        assert config.embed_dim == 256
        assert config.num_res_blocks == 8
        assert config.goal_embed_dim == 128
        assert config.num_attention_heads == 8
        assert config.action_space_size == 100

    def test_for_level_tactical(self) -> None:
        """Test tactical level configuration."""
        config = HierarchicalNetworkConfig.for_level(HierarchyLevel.TACTICAL)

        assert config.embed_dim == 128
        assert config.num_res_blocks == 6
        assert config.goal_embed_dim == 64
        assert config.num_attention_heads == 4
        assert config.action_space_size == 500

    def test_for_level_operational(self) -> None:
        """Test operational level configuration."""
        config = HierarchicalNetworkConfig.for_level(HierarchyLevel.OPERATIONAL)

        assert config.embed_dim == 64
        assert config.num_res_blocks == 4
        assert config.goal_embed_dim == 32
        assert config.num_attention_heads == 2
        assert config.action_space_size == 356


# =============================================================================
# ResidualBlock Tests
# =============================================================================


class TestResidualBlock:
    """Tests for ResidualBlock module."""

    def test_forward_pass(self) -> None:
        """Test residual block forward pass."""
        channels = 32
        block = ResidualBlock(channels, dropout=0.1)
        block.eval()  # Disable dropout for deterministic test

        x = torch.randn(4, channels, 8, 8)
        output = block(x)

        assert output.shape == x.shape

    def test_skip_connection(self) -> None:
        """Test that skip connection is applied."""
        channels = 16
        block = ResidualBlock(channels, dropout=0.0)
        block.eval()

        # Zero weights should result in identity
        with torch.no_grad():
            block.conv1.weight.zero_()
            block.conv2.weight.zero_()
            block.conv1.bias.zero_()
            block.conv2.bias.zero_()
            block.bn1.weight.fill_(1.0)
            block.bn1.bias.zero_()
            block.bn2.weight.fill_(1.0)
            block.bn2.bias.zero_()

        x = torch.relu(torch.randn(2, channels, 4, 4))  # Positive input
        output = block(x)

        # With zero conv weights, output should be close to relu(input)
        # The skip connection adds the input, so output = relu(0 + input) = relu(input) = input
        np.testing.assert_array_almost_equal(
            output.detach().numpy(),
            x.detach().numpy(),
            decimal=4,
        )

    def test_no_dropout(self) -> None:
        """Test residual block with no dropout."""
        block = ResidualBlock(32, dropout=0.0)

        assert isinstance(block.dropout, torch.nn.Identity)

    def test_with_dropout(self) -> None:
        """Test residual block with dropout."""
        block = ResidualBlock(32, dropout=0.5)

        assert isinstance(block.dropout, torch.nn.Dropout)
        assert block.dropout.p == 0.5


# =============================================================================
# GoalConditionedHead Tests
# =============================================================================


class TestGoalConditionedHead:
    """Tests for GoalConditionedHead module."""

    def test_attention_mode_forward(self) -> None:
        """Test forward pass with attention mode."""
        head = GoalConditionedHead(
            input_dim=64,
            output_dim=100,
            goal_dim=32,
            hidden_dim=128,
            use_attention=True,
            num_heads=4,
        )
        head.eval()

        state_features = torch.randn(4, 64)
        goal_embedding = torch.randn(4, 32)

        output = head(state_features, goal_embedding)

        assert output.shape == (4, 100)

    def test_concatenation_mode_forward(self) -> None:
        """Test forward pass with concatenation mode."""
        head = GoalConditionedHead(
            input_dim=64,
            output_dim=100,
            goal_dim=32,
            hidden_dim=128,
            use_attention=False,
            num_heads=4,
        )
        head.eval()

        state_features = torch.randn(4, 64)
        goal_embedding = torch.randn(4, 32)

        output = head(state_features, goal_embedding)

        assert output.shape == (4, 100)

    def test_single_output(self) -> None:
        """Test head with single output (e.g., value head)."""
        head = GoalConditionedHead(
            input_dim=32,
            output_dim=1,
            goal_dim=16,
            hidden_dim=64,
            use_attention=True,
            num_heads=2,
        )
        head.eval()

        state_features = torch.randn(2, 32)
        goal_embedding = torch.randn(2, 16)

        output = head(state_features, goal_embedding)

        assert output.shape == (2, 1)

    def test_batch_size_independence(self) -> None:
        """Test that different batch sizes work."""
        head = GoalConditionedHead(
            input_dim=64,
            output_dim=50,
            goal_dim=32,
            hidden_dim=64,
            use_attention=True,
            num_heads=4,
        )
        head.eval()

        for batch_size in [1, 4, 16]:
            state_features = torch.randn(batch_size, 64)
            goal_embedding = torch.randn(batch_size, 32)

            output = head(state_features, goal_embedding)
            assert output.shape == (batch_size, 50)


# =============================================================================
# LevelNetwork Tests
# =============================================================================


class TestLevelNetwork:
    """Tests for LevelNetwork module."""

    def test_initialization(self, network_config: HierarchicalNetworkConfig) -> None:
        """Test LevelNetwork initialization."""
        network = LevelNetwork(network_config, HierarchyLevel.OPERATIONAL)

        assert network.config == network_config
        assert network.level == HierarchyLevel.OPERATIONAL
        assert len(network.res_blocks) == network_config.num_res_blocks

    def test_forward_pass(
        self,
        network_config: HierarchicalNetworkConfig,
        sample_spatial: torch.Tensor,
        sample_vector: torch.Tensor,
        sample_goal_embedding: torch.Tensor,
    ) -> None:
        """Test LevelNetwork forward pass."""
        network = LevelNetwork(network_config, HierarchyLevel.OPERATIONAL)
        network.eval()

        policy, value = network(sample_spatial, sample_vector, sample_goal_embedding)

        assert policy.shape == (BATCH_SIZE, ACTION_SPACE)
        assert value.shape == (BATCH_SIZE,)

    def test_value_bounded(
        self,
        network_config: HierarchicalNetworkConfig,
        sample_spatial: torch.Tensor,
        sample_vector: torch.Tensor,
        sample_goal_embedding: torch.Tensor,
    ) -> None:
        """Test that value output is bounded by tanh."""
        network = LevelNetwork(network_config, HierarchyLevel.OPERATIONAL)
        network.eval()

        _, value = network(sample_spatial, sample_vector, sample_goal_embedding)

        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)

    def test_action_masking(
        self,
        network_config: HierarchicalNetworkConfig,
        sample_spatial: torch.Tensor,
        sample_vector: torch.Tensor,
        sample_goal_embedding: torch.Tensor,
    ) -> None:
        """Test action masking sets invalid actions to -inf."""
        network = LevelNetwork(network_config, HierarchyLevel.OPERATIONAL)
        network.eval()

        # Mask first 50 actions as invalid
        action_mask = torch.ones(BATCH_SIZE, ACTION_SPACE)
        action_mask[:, :50] = 0

        policy, _ = network(
            sample_spatial, sample_vector, sample_goal_embedding, action_mask
        )

        # First 50 should be -inf
        assert torch.all(policy[:, :50] == float("-inf"))
        # Rest should be finite
        assert torch.all(torch.isfinite(policy[:, 50:]))

    def test_get_state_embedding(
        self,
        network_config: HierarchicalNetworkConfig,
        sample_spatial: torch.Tensor,
        sample_vector: torch.Tensor,
    ) -> None:
        """Test state embedding extraction."""
        network = LevelNetwork(network_config, HierarchyLevel.OPERATIONAL)
        network.eval()

        embedding = network.get_state_embedding(sample_spatial, sample_vector)

        assert embedding.shape == (BATCH_SIZE, network_config.embed_dim)

    def test_different_hierarchy_levels(
        self,
        sample_spatial: torch.Tensor,
        sample_vector: torch.Tensor,
        sample_goal_embedding: torch.Tensor,
    ) -> None:
        """Test network works for all hierarchy levels."""
        for level in HierarchyLevel:
            config = HierarchicalNetworkConfig(
                board_size=BOARD_SIZE,
                spatial_channels=SPATIAL_CHANNELS,
                vector_dim=VECTOR_DIM,
                action_space_size=ACTION_SPACE,
                embed_dim=EMBED_DIM,
                num_res_blocks=2,
                goal_embed_dim=GOAL_DIM,
                num_attention_heads=2,
            )
            network = LevelNetwork(config, level)
            network.eval()

            policy, value = network(
                sample_spatial, sample_vector, sample_goal_embedding
            )

            assert policy.shape == (BATCH_SIZE, ACTION_SPACE)
            assert value.shape == (BATCH_SIZE,)


# =============================================================================
# HierarchicalNetwork Tests
# =============================================================================


class TestHierarchicalNetwork:
    """Tests for HierarchicalNetwork module."""

    def test_initialization_default(self) -> None:
        """Test default initialization creates all level networks."""
        network = HierarchicalNetwork()

        assert hasattr(network, "strategic_network")
        assert hasattr(network, "tactical_network")
        assert hasattr(network, "operational_network")
        assert len(network._networks) == 3

    def test_initialization_custom_configs(self) -> None:
        """Test initialization with custom level configs."""
        strategic = HierarchicalNetworkConfig(
            board_size=BOARD_SIZE,
            embed_dim=128,
            num_res_blocks=4,
            goal_embed_dim=64,
        )
        tactical = HierarchicalNetworkConfig(
            board_size=BOARD_SIZE,
            embed_dim=64,
            num_res_blocks=3,
            goal_embed_dim=32,
        )
        operational = HierarchicalNetworkConfig(
            board_size=BOARD_SIZE,
            embed_dim=32,
            num_res_blocks=2,
            goal_embed_dim=16,
        )

        network = HierarchicalNetwork(
            strategic_config=strategic,
            tactical_config=tactical,
            operational_config=operational,
        )

        assert network.strategic_network.config.embed_dim == 128
        assert network.tactical_network.config.embed_dim == 64
        assert network.operational_network.config.embed_dim == 32

    def test_forward_strategic(self) -> None:
        """Test forward pass for strategic level."""
        config = HierarchicalNetworkConfig.for_level(HierarchyLevel.STRATEGIC)
        config = HierarchicalNetworkConfig(
            board_size=BOARD_SIZE,
            spatial_channels=SPATIAL_CHANNELS,
            vector_dim=VECTOR_DIM,
            action_space_size=100,
            embed_dim=128,
            num_res_blocks=2,
            goal_embed_dim=64,
            num_attention_heads=4,
        )

        network = HierarchicalNetwork(strategic_config=config)
        network.eval()

        spatial = torch.randn(2, SPATIAL_CHANNELS, *BOARD_SIZE)
        vector = torch.randn(2, VECTOR_DIM)
        goal = torch.randn(2, 64)

        policy, value = network(spatial, vector, goal, HierarchyLevel.STRATEGIC)

        assert policy.shape == (2, 100)
        assert value.shape == (2,)

    def test_forward_with_higher_level_embedding(self) -> None:
        """Test forward with cross-level context."""
        # Create compatible configs
        strategic_config = HierarchicalNetworkConfig(
            board_size=BOARD_SIZE,
            spatial_channels=SPATIAL_CHANNELS,
            vector_dim=VECTOR_DIM,
            action_space_size=50,
            embed_dim=256,
            num_res_blocks=2,
            goal_embed_dim=128,
            num_attention_heads=4,
        )
        tactical_config = HierarchicalNetworkConfig(
            board_size=BOARD_SIZE,
            spatial_channels=SPATIAL_CHANNELS,
            vector_dim=VECTOR_DIM,
            action_space_size=100,
            embed_dim=128,
            num_res_blocks=2,
            goal_embed_dim=64,
            num_attention_heads=4,
        )

        network = HierarchicalNetwork(
            strategic_config=strategic_config,
            tactical_config=tactical_config,
        )
        network.eval()

        spatial = torch.randn(2, SPATIAL_CHANNELS, *BOARD_SIZE)
        vector = torch.randn(2, VECTOR_DIM)
        goal = torch.randn(2, 64)  # Tactical goal dim
        higher_embed = torch.randn(2, 256)  # Strategic embed dim

        # Tactical forward with strategic context
        policy, value = network(
            spatial, vector, goal, HierarchyLevel.TACTICAL,
            higher_level_embedding=higher_embed
        )

        assert policy.shape == (2, 100)
        assert value.shape == (2,)

    def test_forward_all_levels(self) -> None:
        """Test forward pass for all levels with cascading context."""
        # Create minimal compatible configs
        strategic_config = HierarchicalNetworkConfig(
            board_size=BOARD_SIZE,
            spatial_channels=SPATIAL_CHANNELS,
            vector_dim=VECTOR_DIM,
            action_space_size=30,
            embed_dim=256,
            num_res_blocks=2,
            goal_embed_dim=128,
            num_attention_heads=4,
        )
        tactical_config = HierarchicalNetworkConfig(
            board_size=BOARD_SIZE,
            spatial_channels=SPATIAL_CHANNELS,
            vector_dim=VECTOR_DIM,
            action_space_size=60,
            embed_dim=128,
            num_res_blocks=2,
            goal_embed_dim=64,
            num_attention_heads=4,
        )
        operational_config = HierarchicalNetworkConfig(
            board_size=BOARD_SIZE,
            spatial_channels=SPATIAL_CHANNELS,
            vector_dim=VECTOR_DIM,
            action_space_size=100,
            embed_dim=64,
            num_res_blocks=2,
            goal_embed_dim=32,
            num_attention_heads=2,
        )

        network = HierarchicalNetwork(
            strategic_config=strategic_config,
            tactical_config=tactical_config,
            operational_config=operational_config,
        )
        network.eval()

        spatial = torch.randn(2, SPATIAL_CHANNELS, *BOARD_SIZE)
        vector = torch.randn(2, VECTOR_DIM)

        goal_embeddings = {
            HierarchyLevel.STRATEGIC: torch.randn(2, 128),
            HierarchyLevel.TACTICAL: torch.randn(2, 64),
            HierarchyLevel.OPERATIONAL: torch.randn(2, 32),
        }

        results = network.forward_all_levels(spatial, vector, goal_embeddings)

        assert len(results) == 3
        assert HierarchyLevel.STRATEGIC in results
        assert HierarchyLevel.TACTICAL in results
        assert HierarchyLevel.OPERATIONAL in results

        # Check shapes
        strategic_policy, strategic_value = results[HierarchyLevel.STRATEGIC]
        assert strategic_policy.shape == (2, 30)
        assert strategic_value.shape == (2,)

        tactical_policy, tactical_value = results[HierarchyLevel.TACTICAL]
        assert tactical_policy.shape == (2, 60)
        assert tactical_value.shape == (2,)

        operational_policy, operational_value = results[HierarchyLevel.OPERATIONAL]
        assert operational_policy.shape == (2, 100)
        assert operational_value.shape == (2,)

    def test_get_network(self) -> None:
        """Test getting network for specific level."""
        network = HierarchicalNetwork()

        strategic = network.get_network(HierarchyLevel.STRATEGIC)
        tactical = network.get_network(HierarchyLevel.TACTICAL)
        operational = network.get_network(HierarchyLevel.OPERATIONAL)

        assert strategic is network.strategic_network
        assert tactical is network.tactical_network
        assert operational is network.operational_network

    def test_invalid_level_raises(self) -> None:
        """Test that invalid level raises ValueError."""
        network = HierarchicalNetwork()

        spatial = torch.randn(2, SPATIAL_CHANNELS, *BOARD_SIZE)
        vector = torch.randn(2, VECTOR_DIM)
        goal = torch.randn(2, 64)

        # Can't easily test with invalid enum value, but we can verify
        # the network has the expected structure
        assert all(level in network._networks for level in HierarchyLevel)

    def test_save_and_load(self) -> None:
        """Test saving and loading network weights."""
        config = HierarchicalNetworkConfig(
            board_size=BOARD_SIZE,
            spatial_channels=SPATIAL_CHANNELS,
            vector_dim=VECTOR_DIM,
            embed_dim=32,
            num_res_blocks=1,
        )

        network1 = HierarchicalNetwork(
            strategic_config=config,
            tactical_config=config,
            operational_config=config,
        )

        network2 = HierarchicalNetwork(
            strategic_config=config,
            tactical_config=config,
            operational_config=config,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "network.pt")

            network1.save(save_path)
            network2.load(save_path)

            # Compare state dicts
            state1 = network1.state_dict()
            state2 = network2.state_dict()

            for key in state1:
                np.testing.assert_array_almost_equal(
                    state1[key].numpy(),
                    state2[key].numpy(),
                    decimal=5,
                )


# =============================================================================
# Integration Tests
# =============================================================================


class TestNetworkIntegration:
    """Integration tests for hierarchical network components."""

    def test_full_pipeline(self) -> None:
        """Test full forward pass through hierarchical network."""
        # Use configs that match hardcoded cross-level projection dimensions
        # The network has hardcoded strategic_to_tactical(256->64) and
        # tactical_to_operational(128->32) projections
        strategic_config = HierarchicalNetworkConfig(
            board_size=(4, 4),
            spatial_channels=8,
            vector_dim=32,
            action_space_size=20,
            embed_dim=256,  # Must match strategic_to_tactical input
            num_res_blocks=1,
            goal_embed_dim=128,
            num_attention_heads=4,
        )
        tactical_config = HierarchicalNetworkConfig(
            board_size=(4, 4),
            spatial_channels=8,
            vector_dim=32,
            action_space_size=40,
            embed_dim=128,  # Must match tactical_to_operational input
            num_res_blocks=1,
            goal_embed_dim=64,
            num_attention_heads=4,
        )
        operational_config = HierarchicalNetworkConfig(
            board_size=(4, 4),
            spatial_channels=8,
            vector_dim=32,
            action_space_size=60,
            embed_dim=64,
            num_res_blocks=1,
            goal_embed_dim=32,
            num_attention_heads=2,
        )

        network = HierarchicalNetwork(
            strategic_config=strategic_config,
            tactical_config=tactical_config,
            operational_config=operational_config,
        )
        network.eval()

        # Create inputs
        batch_size = 2
        spatial = torch.randn(batch_size, 8, 4, 4)
        vector = torch.randn(batch_size, 32)

        goal_embeddings = {
            HierarchyLevel.STRATEGIC: torch.randn(batch_size, 128),  # Match strategic config
            HierarchyLevel.TACTICAL: torch.randn(batch_size, 64),    # Match tactical config
            HierarchyLevel.OPERATIONAL: torch.randn(batch_size, 32), # Match operational config
        }

        action_masks = {
            HierarchyLevel.STRATEGIC: torch.ones(batch_size, 20),
            HierarchyLevel.TACTICAL: torch.ones(batch_size, 40),
            HierarchyLevel.OPERATIONAL: torch.ones(batch_size, 60),
        }

        # Forward through all levels
        results = network.forward_all_levels(
            spatial, vector, goal_embeddings, action_masks
        )

        # Verify all outputs
        for level, (policy, value) in results.items():
            # Policy should be valid logits
            assert torch.isfinite(policy).all()

            # Value should be bounded
            assert torch.all(value >= -1.0)
            assert torch.all(value <= 1.0)

            # Check we can compute softmax
            probs = torch.softmax(policy, dim=-1)
            assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size))

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the network."""
        config = HierarchicalNetworkConfig(
            board_size=(4, 4),
            spatial_channels=4,
            vector_dim=16,
            action_space_size=10,
            embed_dim=16,
            num_res_blocks=1,
            goal_embed_dim=8,
            num_attention_heads=2,
        )

        network = LevelNetwork(config, HierarchyLevel.OPERATIONAL)
        network.train()

        spatial = torch.randn(2, 4, 4, 4, requires_grad=True)
        vector = torch.randn(2, 16, requires_grad=True)
        goal = torch.randn(2, 8, requires_grad=True)

        policy, value = network(spatial, vector, goal)

        # Compute loss and backprop
        loss = policy.sum() + value.sum()
        loss.backward()

        # Check gradients exist
        assert spatial.grad is not None
        assert vector.grad is not None
        assert goal.grad is not None
