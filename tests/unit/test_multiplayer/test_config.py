"""Tests for multiplayer configuration."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from alpha_zero.multiplayer.config import (
    MultiplayerConfig,
    NPlayerMCTSConfig,
    OpponentModelConfig,
    ScoringConfig,
    SearchStrategy,
    TurnManagerConfig,
    TurnMode,
)


class TestNPlayerMCTSConfig:
    """Tests for NPlayerMCTSConfig."""

    def test_default_values(self):
        """Default config has sensible values."""
        config = NPlayerMCTSConfig()
        assert config.num_simulations == 200
        assert config.c_puct == 1.5
        assert config.search_strategy == "max_n"
        assert config.temperature == 1.0

    def test_strategy_property(self):
        """Strategy property returns correct enum."""
        config = NPlayerMCTSConfig(search_strategy="paranoid")
        assert config.strategy == SearchStrategy.PARANOID

    def test_invalid_num_simulations(self):
        """Negative simulations raises ValueError."""
        with pytest.raises(ValueError, match="num_simulations"):
            NPlayerMCTSConfig(num_simulations=0)

    def test_invalid_c_puct(self):
        """Non-positive c_puct raises ValueError."""
        with pytest.raises(ValueError, match="c_puct"):
            NPlayerMCTSConfig(c_puct=-1.0)

    def test_invalid_dirichlet_epsilon(self):
        """Dirichlet epsilon outside [0,1] raises ValueError."""
        with pytest.raises(ValueError, match="dirichlet_epsilon"):
            NPlayerMCTSConfig(dirichlet_epsilon=1.5)

    def test_invalid_search_strategy(self):
        """Unknown search strategy raises ValueError."""
        with pytest.raises(ValueError, match="search_strategy"):
            NPlayerMCTSConfig(search_strategy="unknown")

    def test_all_valid_strategies(self):
        """All SearchStrategy values are accepted."""
        for strategy in SearchStrategy:
            config = NPlayerMCTSConfig(search_strategy=strategy.value)
            assert config.strategy == strategy


class TestScoringConfig:
    """Tests for ScoringConfig."""

    def test_default_values(self):
        config = ScoringConfig()
        assert config.method == "linear_ranking"
        assert config.draw_value == 0.0

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown scoring method"):
            ScoringConfig(method="invalid")

    def test_valid_methods(self):
        for method in ("linear_ranking", "winner_take_all", "proportional"):
            config = ScoringConfig(method=method)
            assert config.method == method


class TestTurnManagerConfig:
    """Tests for TurnManagerConfig."""

    def test_default_values(self):
        config = TurnManagerConfig()
        assert config.mode == "sequential"
        assert config.turn_timeout_seconds == 0.0

    def test_turn_mode_property(self):
        config = TurnManagerConfig(mode="simultaneous")
        assert config.turn_mode == TurnMode.SIMULTANEOUS

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            TurnManagerConfig(mode="invalid")

    def test_all_valid_modes(self):
        for mode in TurnMode:
            config = TurnManagerConfig(mode=mode.value)
            assert config.turn_mode == mode


class TestOpponentModelConfig:
    """Tests for OpponentModelConfig."""

    def test_default_values(self):
        config = OpponentModelConfig()
        assert config.enabled is False
        assert config.history_window == 50
        assert config.decay_rate == 0.95

    def test_invalid_history_window(self):
        with pytest.raises(ValueError, match="history_window"):
            OpponentModelConfig(history_window=0)

    def test_invalid_decay_rate(self):
        with pytest.raises(ValueError, match="decay_rate"):
            OpponentModelConfig(decay_rate=0.0)

        with pytest.raises(ValueError, match="decay_rate"):
            OpponentModelConfig(decay_rate=1.5)


class TestMultiplayerConfig:
    """Tests for MultiplayerConfig."""

    def test_default_values(self):
        config = MultiplayerConfig()
        assert config.num_players == 2
        assert isinstance(config.mcts, NPlayerMCTSConfig)
        assert isinstance(config.scoring, ScoringConfig)

    def test_invalid_num_players(self):
        with pytest.raises(ValueError, match="num_players"):
            MultiplayerConfig(num_players=1)

    def test_total_spatial_channels_2_players(self):
        """2-player config computes correct channel count."""
        config = MultiplayerConfig(num_players=2)
        # base_terrain(3) + 2*(unit_channels(3) + city_channels(1)) + extra(2)
        expected = 3 + 2 * (3 + 1) + 2
        assert config.total_spatial_channels == expected

    def test_total_spatial_channels_4_players(self):
        """4-player config computes correct channel count."""
        config = MultiplayerConfig(num_players=4)
        expected = 3 + 4 * (3 + 1) + 2
        assert config.total_spatial_channels == expected

    def test_total_spatial_channels_scales_with_players(self):
        """Channel count grows linearly with player count."""
        c2 = MultiplayerConfig(num_players=2).total_spatial_channels
        c4 = MultiplayerConfig(num_players=4).total_spatial_channels
        c8 = MultiplayerConfig(num_players=8).total_spatial_channels
        assert c4 > c2
        assert c8 > c4
        # Linear growth: diff should be constant per player
        per_player = (c4 - c2) / 2
        assert abs((c8 - c4) / 4 - per_player) < 0.01

    def test_to_dict(self):
        config = MultiplayerConfig(num_players=3)
        d = config.to_dict()
        assert d["num_players"] == 3
        assert "mcts" in d
        assert "scoring" in d
        assert d["mcts"]["num_simulations"] == 200

    def test_from_dict(self):
        data = {
            "num_players": 4,
            "mcts": {"num_simulations": 100, "search_strategy": "paranoid"},
            "scoring": {"method": "winner_take_all"},
            "turn_manager": {"mode": "simultaneous"},
            "opponent_model": {"enabled": True},
        }
        config = MultiplayerConfig.from_dict(data)
        assert config.num_players == 4
        assert config.mcts.num_simulations == 100
        assert config.mcts.strategy == SearchStrategy.PARANOID
        assert config.scoring.method == "winner_take_all"
        assert config.turn_manager.turn_mode == TurnMode.SIMULTANEOUS
        assert config.opponent_model.enabled is True

    def test_roundtrip_serialization(self):
        original = MultiplayerConfig(num_players=6)
        d = original.to_dict()
        restored = MultiplayerConfig.from_dict(d)
        assert restored.num_players == original.num_players
        assert restored.mcts.num_simulations == original.mcts.num_simulations

    def test_env_overrides(self):
        config = MultiplayerConfig(num_players=2)
        with mock.patch.dict(os.environ, {"CIV6AZ_MP_NUM_PLAYERS": "6"}):
            config.apply_env_overrides()
        assert config.num_players == 6

    def test_env_override_nested(self):
        config = MultiplayerConfig()
        with mock.patch.dict(
            os.environ, {"CIV6AZ_MP_SEARCH_STRATEGY": "paranoid"}
        ):
            config.apply_env_overrides()
        assert config.mcts.search_strategy == "paranoid"

    def test_env_override_bool(self):
        config = MultiplayerConfig()
        with mock.patch.dict(os.environ, {"CIV6AZ_MP_FOG_OF_WAR": "true"}):
            config.apply_env_overrides()
        assert config.fog_of_war is True
