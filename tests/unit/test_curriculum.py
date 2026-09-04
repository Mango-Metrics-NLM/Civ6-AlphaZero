"""Tests for :class:`StockfishCurriculum`."""

from __future__ import annotations

import pytest

pytest.importorskip("chess", reason="requires the 'chess' extra")

from alpha_zero.opponents.stockfish_opponent import (
    STOCKFISH_ELO_MAX,
    STOCKFISH_SKILL_MAX,
    STOCKFISH_SKILL_MIN,
)
from alpha_zero.orchestrator.curriculum import (
    StockfishCurriculum,
    StockfishCurriculumConfig,
)


class TestConfigValidation:
    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="curriculum mode"):
            StockfishCurriculumConfig(mode="bogus")  # type: ignore[arg-type]

    def test_invalid_target_raises(self) -> None:
        with pytest.raises(ValueError, match="curriculum target"):
            StockfishCurriculumConfig(target="bogus")  # type: ignore[arg-type]

    def test_threshold_order_enforced(self) -> None:
        with pytest.raises(ValueError, match="demote_threshold"):
            StockfishCurriculumConfig(promote_threshold=0.3, demote_threshold=0.7)

    def test_window_size_minimum(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            StockfishCurriculumConfig(window_size=0)

    def test_step_every_minimum(self) -> None:
        with pytest.raises(ValueError, match="step_every"):
            StockfishCurriculumConfig(step_every=0)

    def test_initial_value_in_bounds(self) -> None:
        with pytest.raises(ValueError, match="initial_value"):
            StockfishCurriculumConfig(initial_value=STOCKFISH_SKILL_MAX + 1)

    def test_default_skill_bounds_resolved(self) -> None:
        cfg = StockfishCurriculumConfig(target="skill")
        assert cfg.min_value == STOCKFISH_SKILL_MIN
        assert cfg.max_value == STOCKFISH_SKILL_MAX

    def test_default_elo_bounds_resolved(self) -> None:
        cfg = StockfishCurriculumConfig(target="elo", initial_value=1500)
        assert cfg.max_value == STOCKFISH_ELO_MAX


class TestFixedMode:
    def test_value_never_changes(self) -> None:
        c = StockfishCurriculum(StockfishCurriculumConfig(mode="fixed", initial_value=7))
        for win_rate in [0.0, 1.0, 0.5]:
            assert c.update(win_rate) == 7
        assert c.current_value == 7


class TestLinearMode:
    def test_steps_each_iteration(self) -> None:
        c = StockfishCurriculum(
            StockfishCurriculumConfig(
                mode="linear",
                initial_value=0,
                step_size=2,
                step_every=1,
            )
        )
        values = [c.update() for _ in range(3)]
        assert values == [2, 4, 6]

    def test_steps_every_n_iterations(self) -> None:
        c = StockfishCurriculum(
            StockfishCurriculumConfig(
                mode="linear",
                initial_value=0,
                step_size=1,
                step_every=2,
            )
        )
        values = [c.update() for _ in range(5)]
        assert values == [0, 1, 1, 2, 2]

    def test_clamps_at_max(self) -> None:
        c = StockfishCurriculum(
            StockfishCurriculumConfig(
                mode="linear",
                initial_value=STOCKFISH_SKILL_MAX - 1,
                step_size=5,
            )
        )
        v = c.update()
        assert v == STOCKFISH_SKILL_MAX


class TestAdaptiveMode:
    def test_promote_on_high_winrate(self) -> None:
        cfg = StockfishCurriculumConfig(
            mode="adaptive",
            initial_value=5,
            window_size=3,
            promote_threshold=0.7,
            adaptive_step=2,
        )
        c = StockfishCurriculum(cfg)
        for win_rate in [0.8, 0.9, 0.85]:
            value = c.update(win_rate)
        assert value == 7

    def test_demote_on_low_winrate(self) -> None:
        cfg = StockfishCurriculumConfig(
            mode="adaptive",
            initial_value=10,
            window_size=2,
            demote_threshold=0.2,
            adaptive_step=3,
        )
        c = StockfishCurriculum(cfg)
        for win_rate in [0.1, 0.05]:
            value = c.update(win_rate)
        assert value == 7

    def test_no_change_in_middle_band(self) -> None:
        cfg = StockfishCurriculumConfig(
            mode="adaptive",
            initial_value=5,
            window_size=2,
            promote_threshold=0.7,
            demote_threshold=0.3,
        )
        c = StockfishCurriculum(cfg)
        for win_rate in [0.5, 0.55]:
            value = c.update(win_rate)
        assert value == 5

    def test_window_must_fill_before_decision(self) -> None:
        cfg = StockfishCurriculumConfig(
            mode="adaptive",
            initial_value=5,
            window_size=3,
            promote_threshold=0.6,
        )
        c = StockfishCurriculum(cfg)
        # Only two data points; should stay at initial value.
        assert c.update(0.9) == 5
        assert c.update(0.95) == 5

    def test_clamps_at_min_max(self) -> None:
        cfg = StockfishCurriculumConfig(
            mode="adaptive",
            initial_value=STOCKFISH_SKILL_MIN,
            window_size=1,
            demote_threshold=0.5,
            adaptive_step=5,
        )
        c = StockfishCurriculum(cfg)
        value = c.update(0.0)
        assert value == STOCKFISH_SKILL_MIN

    def test_invalid_winrate_raises(self) -> None:
        c = StockfishCurriculum(StockfishCurriculumConfig(mode="adaptive"))
        with pytest.raises(ValueError, match="win_rate"):
            c.update(1.5)


class TestReset:
    def test_reset_clears_state(self) -> None:
        cfg = StockfishCurriculumConfig(
            mode="adaptive",
            initial_value=5,
            window_size=2,
            promote_threshold=0.6,
            adaptive_step=2,
        )
        c = StockfishCurriculum(cfg)
        c.update(0.9)
        c.update(0.9)
        assert c.current_value > 5
        c.reset()
        assert c.current_value == 5

    def test_target_property(self) -> None:
        c = StockfishCurriculum(StockfishCurriculumConfig(target="elo", initial_value=1500))
        assert c.target == "elo"


class TestConfigPropagation:
    def test_environment_config_chess_fields(self) -> None:
        """Verify EnvironmentConfig propagates chess-specific fields."""
        from alpha_zero.orchestrator.config import EnvironmentConfig

        config = EnvironmentConfig.from_dict(
            {
                "type": "chess",
                "gameplay_preset": "chess",
                "chess": {
                    "history_depth": 8,
                    "max_plies": 200,
                    "starting_fen": None,
                },
            }
        )
        assert config.type == "chess"
        assert config.gameplay_preset == "chess"
        assert config.chess_history_depth == 8
        assert config.chess_max_plies == 200
        assert config.chess_starting_fen is None

    def test_environment_config_defaults_unchanged(self) -> None:
        """Backwards-compat: MicroCiv configs ignore new chess fields."""
        from alpha_zero.orchestrator.config import EnvironmentConfig

        config = EnvironmentConfig.from_dict({"type": "microciv", "board_size": 8})
        assert config.gameplay_preset == "default"
        assert config.chess_history_depth == 1
