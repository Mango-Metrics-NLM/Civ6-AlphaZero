"""Unit tests for the N-player training pipeline wiring (gap G9).

Covers the per-player value machinery added to
:mod:`alpha_zero.training_civ6`:

* outcome -> per-player value conversion (``compute_absolute_player_values``)
* relative reordering into the acting player's frame (``relative_value_vector``)
* scoring-method resolution with graceful fallback
* per-player value targets flowing through ``self_play_game`` and
  ``ReplayDataset``
* the N-player value-loss branch in ``train_epoch``
* the scalar-view MCTS adapter (``CurrentPlayerValueView`` / ``_mcts_network``)

Backward compatibility (2-player -> scalar path unchanged) is asserted
throughout so this doubles as a regression guard.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader  # noqa: E402

from alpha_zero.core.civ6_network import MicroCivNetwork  # noqa: E402
from alpha_zero.envs.microciv import MicroCivEnv  # noqa: E402
from alpha_zero.multiplayer.scoring import ScoringMethod  # noqa: E402
from alpha_zero.training_civ6 import (  # noqa: E402
    CurrentPlayerValueView,
    ReplayDataset,
    TrainingConfig,
    TrainingExample,
    _mcts_network,
    _scoring_method_from_str,
    compute_absolute_player_values,
    relative_value_vector,
    run_self_play,
    self_play_game,
    train_epoch,
)

# ---------------------------------------------------------------------------
# Outcome -> per-player value conversion
# ---------------------------------------------------------------------------


class TestComputeAbsolutePlayerValues:
    @pytest.mark.parametrize("num_players", [2, 3, 4, 6])
    def test_winner_take_all_shape_and_winner(self, num_players: int) -> None:
        winner = num_players - 1
        values = compute_absolute_player_values(winner, num_players)
        assert values.shape == (num_players,)
        assert values.dtype == np.float32
        assert values[winner] == pytest.approx(1.0)
        # Every non-winner is a loser under winner-take-all.
        losers = [values[p] for p in range(num_players) if p != winner]
        assert all(v == pytest.approx(-1.0) for v in losers)

    def test_draw_is_all_zero(self) -> None:
        for num_players in (2, 3, 5):
            values = compute_absolute_player_values(None, num_players)
            assert np.allclose(values, 0.0)

    def test_two_player_matches_classic_pm1(self) -> None:
        # The legacy hand-rolled logic: winner -> +1, loser -> -1, draw -> 0.
        assert compute_absolute_player_values(0, 2).tolist() == [1.0, -1.0]
        assert compute_absolute_player_values(1, 2).tolist() == [-1.0, 1.0]
        assert compute_absolute_player_values(None, 2).tolist() == [0.0, 0.0]

    def test_linear_ranking_spreads_values(self) -> None:
        values = compute_absolute_player_values(0, 4, method=ScoringMethod.LINEAR_RANKING)
        # Winner is best; values are bounded to [-1, 1].
        assert values[0] == pytest.approx(1.0)
        assert values.min() >= -1.0 and values.max() <= 1.0


class TestRelativeValueVector:
    @pytest.mark.parametrize("acting", [0, 1, 2, 3])
    def test_index_zero_is_acting_player(self, acting: int) -> None:
        absolute = compute_absolute_player_values(winner=2, num_players=4)
        relative = relative_value_vector(absolute, acting, 4)
        assert relative.shape == (4,)
        assert relative[0] == pytest.approx(absolute[acting])

    def test_is_a_rotation_of_absolute(self) -> None:
        absolute = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        relative = relative_value_vector(absolute, acting_player=1, num_players=4)
        assert relative.tolist() == pytest.approx([0.2, 0.3, 0.4, 0.1])


class TestScoringMethodResolution:
    @pytest.mark.parametrize("name", [m.value for m in ScoringMethod])
    def test_valid_names_resolve(self, name: str) -> None:
        assert _scoring_method_from_str(name) == ScoringMethod(name)

    def test_unknown_falls_back_to_winner_take_all(self) -> None:
        assert _scoring_method_from_str("nonsense-method") == ScoringMethod.WINNER_TAKE_ALL


# ---------------------------------------------------------------------------
# TrainingExample / ReplayDataset
# ---------------------------------------------------------------------------


def _example(value: float, value_vector: np.ndarray | None) -> TrainingExample:
    return TrainingExample(
        spatial=np.zeros((3, 4, 4), dtype=np.float32),
        vector=np.zeros((8,), dtype=np.float32),
        action_mask=np.ones((5,), dtype=np.float32),
        policy=np.ones((5,), dtype=np.float32) / 5.0,
        value=value,
        value_vector=value_vector,
    )


class TestTrainingExampleAndDataset:
    def test_value_vector_defaults_to_none(self) -> None:
        ex = _example(1.0, None)
        assert ex.value_vector is None

    def test_dataset_yields_scalar_when_no_vector(self) -> None:
        ds = ReplayDataset([_example(1.0, None)])
        item = ds[0]
        assert item["value"].ndim == 0  # scalar

    def test_dataset_yields_vector_when_present(self) -> None:
        vec = np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        ds = ReplayDataset([_example(1.0, vec)])
        item = ds[0]
        assert item["value"].shape == (4,)

    def test_scalar_examples_collate_to_1d_batch(self) -> None:
        ds = ReplayDataset([_example(1.0, None) for _ in range(4)])
        batch = next(iter(DataLoader(ds, batch_size=4)))
        assert batch["value"].shape == (4,)

    def test_vector_examples_collate_to_2d_batch(self) -> None:
        vec = np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        ds = ReplayDataset([_example(1.0, vec) for _ in range(4)])
        batch = next(iter(DataLoader(ds, batch_size=4)))
        assert batch["value"].shape == (4, 4)


# ---------------------------------------------------------------------------
# self_play_game — per-player targets from a real env
# ---------------------------------------------------------------------------


class TestSelfPlayGameValueTargets:
    def _play(self, num_players: int) -> list[TrainingExample]:
        from alpha_zero.core.civ6_mcts import SimpleMCTS

        cfg = TrainingConfig(board_size=6, num_players=num_players, num_simulations=3, max_turns=10)
        env = MicroCivEnv(board_size=6, max_turns=10, num_players=num_players)
        mcts = SimpleMCTS(num_simulations=3, c_puct=1.5)
        return self_play_game(env, mcts, cfg)

    def test_two_player_has_no_value_vector(self) -> None:
        examples = self._play(2)
        assert examples, "self-play produced no examples"
        assert all(ex.value_vector is None for ex in examples)
        # Scalar values are the classic +1 / -1 / 0 signal.
        assert all(ex.value in (1.0, -1.0, 0.0) for ex in examples)

    @pytest.mark.parametrize("num_players", [3, 4])
    def test_nplayer_populates_relative_vector(self, num_players: int) -> None:
        examples = self._play(num_players)
        assert examples, "self-play produced no examples"
        for ex in examples:
            assert ex.value_vector is not None
            assert ex.value_vector.shape == (num_players,)
            # value_vector[0] is the acting player's own outcome == scalar value.
            assert ex.value_vector[0] == pytest.approx(ex.value)


# ---------------------------------------------------------------------------
# train_epoch — N-player value-loss branch
# ---------------------------------------------------------------------------


class TestTrainEpochNPlayer:
    def _make(self, num_players: int, spatial_channels: int) -> tuple:
        board = 6
        action_size = 32
        net = MicroCivNetwork(
            board_size=board,
            action_space_size=action_size,
            num_res_blocks=1,
            num_filters=8,
            spatial_channels=spatial_channels,
            num_players=num_players,
        )
        examples = []
        rng = np.random.default_rng(0)
        for _ in range(6):
            av = compute_absolute_player_values(
                winner=int(rng.integers(num_players)), num_players=num_players
            )
            vec = relative_value_vector(av, 0, num_players) if num_players > 2 else None
            examples.append(
                TrainingExample(
                    spatial=rng.standard_normal((spatial_channels, board, board)).astype(
                        np.float32
                    ),
                    vector=rng.standard_normal(64).astype(np.float32),
                    action_mask=np.ones(action_size, dtype=np.float32),
                    policy=(np.ones(action_size, dtype=np.float32) / action_size),
                    value=float(av[0]),
                    value_vector=vec,
                )
            )
        dl = DataLoader(ReplayDataset(examples), batch_size=3)
        return net, dl

    def test_nplayer_train_epoch_produces_finite_losses(self) -> None:
        num_players = 4
        spatial_channels = 3 + 4 * num_players + 2  # dynamic encoder channel count
        cfg = TrainingConfig(board_size=6, num_players=num_players, batch_size=3, device="cpu")
        net, dl = self._make(num_players, spatial_channels)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        losses = train_epoch(net, opt, dl, cfg)
        assert np.isfinite(losses["total_loss"])
        assert np.isfinite(losses["value_loss"])
        assert losses["value_loss"] >= 0.0

    def test_nplayer_network_emits_per_player_value(self) -> None:
        num_players = 4
        spatial_channels = 3 + 4 * num_players + 2
        net, dl = self._make(num_players, spatial_channels)
        batch = next(iter(dl))
        _, value = net(batch["spatial"], batch["vector"], batch["action_mask"])
        assert value.shape == (batch["spatial"].shape[0], num_players)


# ---------------------------------------------------------------------------
# Scalar-view MCTS adapter
# ---------------------------------------------------------------------------


class TestCurrentPlayerValueView:
    def test_reduces_per_player_value_to_current_player(self) -> None:
        num_players = 4
        spatial_channels = 3 + 4 * num_players + 2
        net = MicroCivNetwork(
            board_size=6,
            action_space_size=16,
            num_res_blocks=1,
            num_filters=8,
            spatial_channels=spatial_channels,
            num_players=num_players,
        )
        view = CurrentPlayerValueView(net)
        spatial = torch.zeros((1, spatial_channels, 6, 6))
        vector = torch.zeros((1, 64))
        mask = torch.ones((1, 16))
        _, full_value = net(spatial, vector, mask)
        _, scalar_value = view.predict(spatial, vector, mask)
        assert full_value.shape == (1, num_players)
        assert scalar_value.shape == (1,)
        # The scalar view exposes the current (relative index 0) player's value.
        assert scalar_value.item() == pytest.approx(full_value[0, 0].item(), abs=1e-5)

    def test_scalar_value_passes_through(self) -> None:
        # A 2-player (scalar-head) network is not double-reduced.
        net = MicroCivNetwork(board_size=6, action_space_size=16, num_res_blocks=1, num_filters=8)
        view = CurrentPlayerValueView(net)
        spatial = torch.zeros((1, 13, 6, 6))
        vector = torch.zeros((1, 64))
        mask = torch.ones((1, 16))
        _, value = view.predict(spatial, vector, mask)
        assert value.shape == (1,)

    def test_shares_parameters_with_wrapped_network(self) -> None:
        net = MicroCivNetwork(board_size=6, action_space_size=16, num_res_blocks=1, num_filters=8)
        view = CurrentPlayerValueView(net)
        # Same tensor objects -> weights are shared, not copied.
        wrapped = dict(view.network.named_parameters())
        for name, param in net.named_parameters():
            assert wrapped[name] is param


class TestMctsNetworkSelection:
    def test_two_player_returns_network_unchanged(self) -> None:
        net = MicroCivNetwork(board_size=6, action_space_size=16, num_res_blocks=1, num_filters=8)
        cfg = TrainingConfig(num_players=2, device="cpu")
        assert _mcts_network(net, cfg) is net

    def test_nplayer_wraps_in_scalar_view(self) -> None:
        spatial_channels = 3 + 4 * 4 + 2
        net = MicroCivNetwork(
            board_size=6,
            action_space_size=16,
            num_res_blocks=1,
            num_filters=8,
            spatial_channels=spatial_channels,
            num_players=4,
        )
        cfg = TrainingConfig(num_players=4, device="cpu")
        wrapped = _mcts_network(net, cfg)
        assert isinstance(wrapped, CurrentPlayerValueView)
        assert wrapped.network is net


# ---------------------------------------------------------------------------
# End-to-end: run_self_play drives the N-player path
# ---------------------------------------------------------------------------


class TestRunSelfPlayNPlayer:
    def test_run_self_play_produces_nplayer_examples(self) -> None:
        cfg = TrainingConfig(
            board_size=6, num_players=4, num_simulations=3, max_turns=8, device="cpu"
        )
        # use_simple_mcts avoids needing a trained network for the smoke path.
        examples = run_self_play(
            network=None,  # type: ignore[arg-type]
            config=cfg,
            num_games=1,
            use_simple_mcts=True,
        )
        assert examples
        assert all(ex.value_vector is not None for ex in examples)
        assert all(ex.value_vector.shape == (4,) for ex in examples)
