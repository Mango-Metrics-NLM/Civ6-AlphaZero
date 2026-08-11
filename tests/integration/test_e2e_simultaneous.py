"""End-to-end tests for DUCT (Decoupled UCT) simultaneous-turn support.

DUCT is a Phase 23 feature that may not yet be implemented. All tests
in this module are guarded with skipif markers so they degrade gracefully
when DUCT imports are unavailable.

Also includes tests for simultaneous turn flow with the existing
TurnManager infrastructure.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.multiplayer.config import TurnManagerConfig, TurnMode
from alpha_zero.multiplayer.turn_manager import TurnManager

# ---------------------------------------------------------------------------
# DUCT availability check
# ---------------------------------------------------------------------------

try:
    from alpha_zero.multiplayer.simultaneous_mcts import (  # type: ignore[import-untyped]
        DecoupledUCT,
        SimultaneousMCTSConfig,
    )

    DUCT_AVAILABLE = True
except ImportError:
    DUCT_AVAILABLE = False


# ===========================================================================
# TestDUCTEndToEnd
# ===========================================================================


@pytest.mark.integration
@pytest.mark.skipif(not DUCT_AVAILABLE, reason="DUCT (DecoupledUCT) not yet implemented")
class TestDUCTEndToEnd:
    """DUCT-specific end-to-end tests."""

    def test_duct_search_returns_per_player_probs(self) -> None:
        """DecoupledUCT.search with 4-player env returns dict mapping player -> probs."""
        from alpha_zero.core.civ6_network import MicroCivNetwork
        from alpha_zero.envs.microciv import MicroCivEnv
        from tests.test_constants import TestDefaults, microciv_channels

        num_players = 4
        channels = microciv_channels(num_players)
        board = TestDefaults.SMALL_BOARD_SIZE

        env = MicroCivEnv(board_size=board, max_turns=50, num_players=num_players, seed=42)
        net = MicroCivNetwork(
            board_size=board,
            action_space_size=int(env.action_space.n),
            num_res_blocks=2,
            num_filters=32,
            spatial_channels=channels,
            num_players=num_players,
        )
        net.eval()
        env.reset()

        config = SimultaneousMCTSConfig(num_simulations=10)
        duct = DecoupledUCT(
            config=config, num_players=num_players, action_space_size=int(env.action_space.n)
        )
        active_players = list(range(num_players))
        result = duct.search(env, net, active_players)

        # Result should map player_id -> action probability array
        assert isinstance(result, dict)
        assert len(result) == num_players
        for _, probs in result.items():
            assert isinstance(probs, np.ndarray)
            assert probs.shape == (env.action_space.n,)

    def test_duct_joint_actions_valid(self) -> None:
        """Verify all per-player actions from DUCT are valid."""
        from alpha_zero.core.civ6_network import MicroCivNetwork
        from alpha_zero.envs.microciv import MicroCivEnv
        from tests.test_constants import TestDefaults, microciv_channels

        num_players = 4
        channels = microciv_channels(num_players)
        board = TestDefaults.SMALL_BOARD_SIZE

        env = MicroCivEnv(board_size=board, max_turns=50, num_players=num_players, seed=42)
        net = MicroCivNetwork(
            board_size=board,
            action_space_size=int(env.action_space.n),
            num_res_blocks=2,
            num_filters=32,
            spatial_channels=channels,
            num_players=num_players,
        )
        net.eval()
        env.reset()

        config = SimultaneousMCTSConfig(num_simulations=10)
        duct = DecoupledUCT(
            config=config, num_players=num_players, action_space_size=int(env.action_space.n)
        )
        active_players = list(range(num_players))
        result = duct.search(env, net, active_players)

        for _, probs in result.items():
            action = int(np.argmax(probs))
            # Action should be within the valid action space
            assert action < int(env.action_space.n)

    def test_duct_probs_sum_to_one(self) -> None:
        """Verify each player's action probs from DUCT sum to ~1.0."""
        from alpha_zero.core.civ6_network import MicroCivNetwork
        from alpha_zero.envs.microciv import MicroCivEnv
        from tests.test_constants import TestDefaults, microciv_channels

        num_players = 4
        channels = microciv_channels(num_players)
        board = TestDefaults.SMALL_BOARD_SIZE

        env = MicroCivEnv(board_size=board, max_turns=50, num_players=num_players, seed=42)
        net = MicroCivNetwork(
            board_size=board,
            action_space_size=int(env.action_space.n),
            num_res_blocks=2,
            num_filters=32,
            spatial_channels=channels,
            num_players=num_players,
        )
        net.eval()
        env.reset()

        config = SimultaneousMCTSConfig(num_simulations=10)
        duct = DecoupledUCT(
            config=config, num_players=num_players, action_space_size=int(env.action_space.n)
        )
        active_players = list(range(num_players))
        result = duct.search(env, net, active_players)

        for player_id, probs in result.items():
            assert np.all(probs >= 0), f"Player {player_id}: negative probabilities"
            assert abs(probs.sum() - 1.0) < 1e-4, (
                f"Player {player_id}: probs sum to {probs.sum()}, not 1.0"
            )


# ===========================================================================
# TestSimultaneousTurnFlow
# ===========================================================================


@pytest.mark.integration
class TestSimultaneousTurnFlow:
    """Simultaneous turn management with action resolution."""

    def test_turn_manager_simultaneous_with_resolver(self) -> None:
        """TurnManager(simultaneous) activates all players and collects actions."""
        num_players = 4
        config = TurnManagerConfig(mode=TurnMode.SIMULTANEOUS.value)
        tm = TurnManager(config, num_players=num_players)

        # Begin turn: all players should be active
        state = tm.begin_turn()
        active = tm.get_active_players()
        assert len(active) == num_players
        assert set(active) == set(range(num_players))

        # Submit actions for all players
        for player_id in range(num_players):
            success = tm.submit_action(player_id, action=player_id * 10)
            assert success, f"Player {player_id} action submission failed"

        # All actions should be submitted
        assert tm.all_actions_submitted()

        # Verify action values
        assert state.actions_submitted[0] == 0
        assert state.actions_submitted[1] == 10
        assert state.actions_submitted[2] == 20
        assert state.actions_submitted[3] == 30

        tm.end_turn()

        # Turn number should have advanced
        assert tm.turn_number == 1

    def test_graceful_degradation_without_duct(self) -> None:
        """When DUCT is not available, system uses sequential fallback."""
        # The TurnManager should work in sequential mode regardless of DUCT
        num_players = 4
        config = TurnManagerConfig(mode=TurnMode.SEQUENTIAL.value)
        tm = TurnManager(config, num_players=num_players)

        # Begin turn: only one player should be active
        tm.begin_turn()
        active = tm.get_active_players()
        assert len(active) == 1
        current_player = active[0]

        # Submit action for the active player
        success = tm.submit_action(current_player, action=42)
        assert success

        # Non-active players should be rejected
        other_player = (current_player + 1) % num_players
        rejected = tm.submit_action(other_player, action=99)
        assert not rejected

        tm.end_turn()

        # Next turn should have a different sequential player
        tm.begin_turn()
        new_active = tm.get_active_players()
        assert len(new_active) == 1
        assert new_active[0] != current_player
