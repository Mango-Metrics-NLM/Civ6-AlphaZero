"""Tests for multiplayer protocol definitions and public API exports.

Verifies that:
- Protocol types are runtime-checkable
- Mock implementations satisfy protocol contracts
- All public exports from __init__.py are importable and correct
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from alpha_zero.multiplayer import (
    GameConnector,
    GameConnectorConfig,
    MockGameConnector,
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
    OpponentModel,
    OpponentModelConfig,
    ScoringConfig,
    ScoringMethod,
    SearchStrategy,
    TurnManager,
    TurnManagerConfig,
    TurnMode,
    UncivMultiplayerConnector,
    compute_nplayer_value_loss,
    compute_player_values,
    compute_rankings,
    create_game_connector,
    extract_game_result,
)
from alpha_zero.multiplayer.protocols import OpponentModelProtocol


class SimpleActionSpace:
    """Minimal action space for protocol tests."""

    def __init__(self, n: int = 4):
        self.n = n


class MockEnvForProtocol:
    """Minimal environment that should satisfy MultiplayerEnvironment protocol."""

    def __init__(self, num_players: int = 3):
        self.action_space = SimpleActionSpace(4)
        self.current_player = 0
        self.num_players = num_players

    def clone(self) -> MockEnvForProtocol:
        cloned = MockEnvForProtocol(self.num_players)
        cloned.current_player = self.current_player
        return cloned

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.current_player = (self.current_player + 1) % self.num_players
        return {}, 0.0, False, False, {}

    def get_observation(self) -> dict[str, np.ndarray]:
        return {
            "spatial": np.zeros((13, 4, 4), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
            "action_mask": np.ones(4, dtype=np.float32),
        }

    def get_info(self) -> dict[str, Any]:
        return {}

    def get_valid_actions(self) -> np.ndarray:
        return np.ones(4, dtype=np.float32)


class MockNetForProtocol:
    """Minimal network that should satisfy MultiplayerNetwork protocol."""

    def __init__(self, num_players: int = 3):
        self._num_players = num_players

    @property
    def num_players(self) -> int:
        return self._num_players

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = spatial.shape[0]
        policy = torch.ones(b, 4) / 4
        values = torch.zeros(b, self._num_players)
        return policy, values

    def eval(self) -> None:
        pass


class TestMultiplayerEnvironmentProtocol:
    """Tests for MultiplayerEnvironment protocol."""

    def test_protocol_is_runtime_checkable(self):
        """MultiplayerEnvironment supports isinstance checks."""
        env = MockEnvForProtocol()
        assert isinstance(env, MultiplayerEnvironment)

    def test_protocol_rejects_non_conforming(self):
        """Non-conforming objects don't satisfy protocol."""

        class NotAnEnv:
            pass

        assert not isinstance(NotAnEnv(), MultiplayerEnvironment)

    def test_protocol_requires_all_attributes(self):
        """Missing attributes cause protocol check to fail."""

        class MissingNumPlayers:
            action_space = SimpleActionSpace()
            current_player = 0
            # Missing: num_players

            def clone(self):
                pass

            def step(self, action):
                pass

            def get_observation(self):
                pass

            def get_info(self):
                pass

            def get_valid_actions(self):
                pass

        assert not isinstance(MissingNumPlayers(), MultiplayerEnvironment)


class TestMultiplayerNetworkProtocol:
    """Tests for MultiplayerNetwork protocol."""

    def test_protocol_is_runtime_checkable(self):
        """MultiplayerNetwork supports isinstance checks."""
        net = MockNetForProtocol()
        assert isinstance(net, MultiplayerNetwork)

    def test_protocol_rejects_non_conforming(self):
        """Non-conforming objects don't satisfy protocol."""

        class NotANetwork:
            pass

        assert not isinstance(NotANetwork(), MultiplayerNetwork)


class TestOpponentModelProtocol:
    """Tests for OpponentModelProtocol."""

    def test_protocol_is_runtime_checkable(self):
        """OpponentModelProtocol supports isinstance checks."""
        config = OpponentModelConfig(enabled=True, history_window=10)
        model = OpponentModel(config, num_players=3, action_space_size=10)
        assert isinstance(model, OpponentModelProtocol)


class TestPublicExports:
    """Verify all public exports from multiplayer.__init__."""

    def test_all_exports_exist(self):
        """Every item in __all__ is importable."""
        import alpha_zero.multiplayer as mp

        for name in mp.__all__:
            assert hasattr(mp, name), f"{name} listed in __all__ but not importable"

    def test_config_classes_importable(self):
        """Config dataclasses are importable from top level."""
        assert MultiplayerConfig is not None
        assert NPlayerMCTSConfig is not None
        assert ScoringConfig is not None
        assert TurnManagerConfig is not None
        assert OpponentModelConfig is not None
        assert NPlayerNetworkConfig is not None
        assert GameConnectorConfig is not None

    def test_enum_types_importable(self):
        """Enum types are importable from top level."""
        assert SearchStrategy.MAX_N is not None
        assert SearchStrategy.PARANOID is not None
        assert SearchStrategy.BEST_REPLY is not None
        assert TurnMode.SEQUENTIAL is not None
        assert TurnMode.SIMULTANEOUS is not None
        assert TurnMode.DYNAMIC is not None
        assert ScoringMethod.LINEAR_RANKING is not None

    def test_core_classes_importable(self):
        """Core classes are importable from top level."""
        assert NPlayerMCTS is not None
        assert NPlayerMCTSNode is not None
        assert NPlayerValueHead is not None
        assert NPlayerNetworkAdapter is not None
        assert NPlayerEnvAdapter is not None
        assert TurnManager is not None
        assert OpponentModel is not None

    def test_game_connectors_importable(self):
        """Game connectors and factory are importable."""
        assert GameConnector is not None
        assert MockGameConnector is not None
        assert UncivMultiplayerConnector is not None
        assert create_game_connector is not None

    def test_functions_importable(self):
        """Utility functions are importable from top level."""
        assert callable(compute_rankings)
        assert callable(compute_player_values)
        assert callable(extract_game_result)
        assert callable(compute_nplayer_value_loss)
        assert callable(create_game_connector)

    def test_protocol_types_importable(self):
        """Protocol types are importable from top level."""
        assert MultiplayerEnvironment is not None
        assert MultiplayerNetwork is not None
