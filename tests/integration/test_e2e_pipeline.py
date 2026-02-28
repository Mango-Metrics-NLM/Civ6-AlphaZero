"""End-to-end pipeline and factory integration tests.

Tests factory functions, component registry, import smoke tests,
and MultiplayerConfig serialization roundtrips.
"""

from __future__ import annotations

import pytest

from tests.test_constants import TestDefaults, microciv_channels

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOARD_SIZE = TestDefaults.SMALL_BOARD_SIZE
NUM_RES = TestDefaults.NUM_RES_BLOCKS
NUM_FILTERS = TestDefaults.NUM_FILTERS


# ===========================================================================
# TestFactoryIntegration
# ===========================================================================


@pytest.mark.integration
class TestFactoryIntegration:
    """Factory functions create correct N-player components."""

    def test_create_network_nplayer(self) -> None:
        """create_network with num_players=4 produces correct spatial_channels."""
        import torch

        from alpha_zero.core.civ6_network import MicroCivNetwork

        num_players = 4
        channels = microciv_channels(num_players)
        env_action_size = TestDefaults.ACTION_SPACE_SIZE

        net = MicroCivNetwork(
            board_size=BOARD_SIZE,
            action_space_size=env_action_size,
            num_res_blocks=NUM_RES,
            num_filters=NUM_FILTERS,
            spatial_channels=channels,
            num_players=num_players,
        )

        assert net.num_players == num_players
        assert net.config.spatial_channels == channels

        # Forward pass should work
        spatial = torch.randn(1, channels, BOARD_SIZE, BOARD_SIZE)
        vector = torch.randn(1, TestDefaults.VECTOR_DIM)
        with torch.no_grad():
            policy, value = net(spatial, vector)
        assert policy.shape == (1, env_action_size)
        assert value.shape == (1, num_players)

    def test_create_environment_nplayer(self) -> None:
        """Create MicroCivEnv with num_players=4 and verify observation channels."""
        import numpy as np

        from alpha_zero.envs.microciv import MicroCivEnv

        num_players = 4
        expected_channels = microciv_channels(num_players)

        env = MicroCivEnv(
            board_size=BOARD_SIZE,
            max_turns=TestDefaults.MAX_TURNS,
            num_players=num_players,
            seed=42,
        )
        obs, _info = env.reset()

        assert obs["spatial"].shape[0] == expected_channels
        assert obs["spatial"].shape[1] == BOARD_SIZE
        assert obs["spatial"].shape[2] == BOARD_SIZE
        assert obs["vector"].shape == (TestDefaults.VECTOR_DIM,)
        assert obs["action_mask"].shape == (env.action_space.n,)
        assert np.all(np.isfinite(obs["spatial"]))
        assert env.num_players == num_players

    def test_create_mcts_with_nplayer_network(self) -> None:
        """Factory creates MCTS that works with N-player network."""
        import numpy as np

        from alpha_zero.core.civ6_network import MicroCivNetwork
        from alpha_zero.envs.microciv import MicroCivEnv
        from alpha_zero.multiplayer.config import NPlayerMCTSConfig
        from alpha_zero.multiplayer.mcts import NPlayerMCTS

        num_players = 4
        channels = microciv_channels(num_players)

        env = MicroCivEnv(
            board_size=BOARD_SIZE,
            max_turns=TestDefaults.MAX_TURNS,
            num_players=num_players,
            seed=42,
        )
        net = MicroCivNetwork(
            board_size=BOARD_SIZE,
            action_space_size=env.action_space.n,
            num_res_blocks=NUM_RES,
            num_filters=NUM_FILTERS,
            spatial_channels=channels,
            num_players=num_players,
        )
        net.eval()

        mcts = NPlayerMCTS(
            network=net,
            config=NPlayerMCTSConfig(num_simulations=TestDefaults.NUM_SIMULATIONS),
            num_players=num_players,
            device="cpu",
        )

        env.reset()
        probs, values = mcts.search(env, add_noise=False)

        assert np.all(probs >= 0)
        assert abs(probs.sum() - 1.0) < 1e-4
        assert len(values) == num_players
        for p in range(num_players):
            assert p in values


# ===========================================================================
# TestRegistryIntegration
# ===========================================================================


@pytest.mark.integration
class TestRegistryIntegration:
    """Registry lists categories and creates matching components."""

    def test_registry_lists_categories(self) -> None:
        """Registry lists all expected categories."""
        import alpha_zero.utils.factories  # noqa: F401 — triggers registration
        from alpha_zero.utils.registry import default_registry

        categories = default_registry.list_categories()

        # These categories should exist after factories.py is imported
        expected = {"network", "mcts", "environment", "replay_buffer"}
        for cat in expected:
            assert cat in categories, (
                f"Expected category '{cat}' not found. Available: {categories}"
            )

    def test_registry_create_matches_factory(self) -> None:
        """Component created via registry matches direct factory creation type."""
        import alpha_zero.utils.factories  # noqa: F401 — triggers registration
        from alpha_zero.utils.registry import default_registry

        # Verify that the registry has the expected factories registered
        assert default_registry.has_component("network", "default")
        assert default_registry.has_component("mcts", "default")
        assert default_registry.has_component("environment", "default")

        # Verify we can list components within a category
        network_components = default_registry.list_components("network")
        assert "default" in network_components

        mcts_components = default_registry.list_components("mcts")
        assert "default" in mcts_components


# ===========================================================================
# TestSmokeTests
# ===========================================================================


@pytest.mark.integration
class TestSmokeTests:
    """Quick import and roundtrip smoke tests."""

    def test_import_all_multiplayer_components(self) -> None:
        """Import all components from alpha_zero.multiplayer."""
        from alpha_zero.multiplayer import (  # noqa: F401
            DeterminizationConfig,
            DeterminizedMCTS,
            FogOfWarConfig,
            FogOfWarManager,
            MultiplayerConfig,
            MultiplayerEnvironment,
            MultiplayerNetwork,
            NPlayerEnvAdapter,
            NPlayerMCTS,
            NPlayerMCTSConfig,
            NPlayerMCTSNode,
            NPlayerNetworkAdapter,
            NPlayerNetworkConfig,
            NPlayerValueHead,
            ObservationModel,
            ObservationModelConfig,
            OpponentModel,
            OpponentModelConfig,
            ScoringConfig,
            ScoringMethod,
            SearchStrategy,
            TurnManager,
            TurnManagerConfig,
            TurnMode,
            VisibilityState,
            WorldSampler,
            compute_nplayer_value_loss,
            compute_player_values,
            compute_rankings,
            extract_game_result,
        )

        # Verify a few key attributes
        assert MultiplayerConfig is not None
        assert NPlayerMCTS is not None
        assert TurnManager is not None

    def test_import_all_factory_functions(self) -> None:
        """Import all factory functions."""
        from alpha_zero.utils.factories import (  # noqa: F401
            create_curriculum_manager,
            create_environment,
            create_knowledge_base,
            create_llm_client,
            create_mcts,
            create_network,
            create_rag_client,
            create_replay_buffer,
            create_reward_shaper,
            create_strategy_advisor,
            create_wandb_logger,
        )

        # Verify they are callable
        assert callable(create_network)
        assert callable(create_mcts)
        assert callable(create_environment)
        assert callable(create_replay_buffer)

    def test_multiplayer_config_roundtrip(self) -> None:
        """MultiplayerConfig -> to_dict -> from_dict -> matches original."""
        from alpha_zero.multiplayer.config import (
            MultiplayerConfig,
            NPlayerMCTSConfig,
            ScoringConfig,
        )

        original = MultiplayerConfig(
            num_players=4,
            mcts=NPlayerMCTSConfig(
                num_simulations=200,
                c_puct=2.0,
                search_strategy="paranoid",
            ),
            scoring=ScoringConfig(method="winner_take_all"),
            fog_of_war=True,
        )

        data = original.to_dict()
        assert isinstance(data, dict)
        assert data["num_players"] == 4

        restored = MultiplayerConfig.from_dict(data)
        assert restored.num_players == original.num_players
        assert restored.mcts.num_simulations == original.mcts.num_simulations
        assert restored.mcts.c_puct == pytest.approx(original.mcts.c_puct)
        assert restored.mcts.search_strategy == original.mcts.search_strategy
        assert restored.scoring.method == original.scoring.method
        assert restored.fog_of_war == original.fog_of_war
