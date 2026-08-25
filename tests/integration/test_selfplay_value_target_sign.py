"""Regression tests for per-player value-target signing in cloud self-play.

Before this was fixed, all three self-play paths in
``alpha_zero/cloud/gcp/selfplay_worker.py`` broadcast a single scalar to every
position in a trajectory::

    final_value = reward
    trajectory["rewards"] = [final_value] * num_steps

``reward`` is relative to whoever was ``current_player`` at the terminal action,
so in a two-player zero-sum game roughly half of every trajectory carried the
wrong sign. The local pipeline (``alpha_zero/training_civ6.py``) has always
signed per player; these tests pin the cloud path to the same convention.

The central invariant is winner-free and was provably violated by the old code:
within one trajectory, every position belonging to the same player must carry the
same value, and the distinct values must be either ``{0.0}`` (a draw) or
``{+1.0, -1.0}`` (a decisive game).
"""

from __future__ import annotations

import pytest

from alpha_zero.cloud.gcp.selfplay_worker import outcome_value_for_player

_BOARD_SIZE = 8
_MAX_TURNS = 15
_MCTS_SIMS = 2
_GUMBEL_ACTIONS = 4


@pytest.fixture
def small_model():
    """A deliberately tiny network -- these tests are about labels, not strength."""
    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

    model = Civ6TransformerNetwork(
        config=TransformerConfig(
            spatial_channels=13,
            spatial_size=_BOARD_SIZE,
            vector_dim=64,
            action_space_size=376,
            num_layers=1,
            embed_dim=32,
        )
    )
    model.eval()
    return model


def assert_value_targets_are_zero_sum(trajectory: dict) -> None:
    """Assert the winner-free invariant on one trajectory."""
    players = trajectory["players"]
    rewards = trajectory["rewards"]
    assert len(players) == len(rewards) == len(trajectory["spatial"])

    by_player: dict[int, set[float]] = {}
    for player, reward in zip(players, rewards, strict=True):
        by_player.setdefault(player, set()).add(float(reward))

    for player, values in by_player.items():
        assert len(values) == 1, f"player {player} has inconsistent value targets: {values}"

    distinct = {next(iter(v)) for v in by_player.values()}
    assert distinct in ({0.0}, {1.0, -1.0}, {1.0}, {-1.0}), (
        f"value targets must be a draw or zero-sum, got {distinct}"
    )

    # A decisive game observed from both seats must contain both signs. This is
    # the exact assertion the pre-fix implementation could never satisfy.
    if len(by_player) > 1 and distinct != {0.0}:
        assert distinct == {1.0, -1.0}, (
            f"decisive game seen from {len(by_player)} seats must carry both signs, got {distinct}"
        )


class TestOutcomeValueForPlayer:
    def test_winner_loser_draw(self) -> None:
        assert outcome_value_for_player(0, 0) == 1.0
        assert outcome_value_for_player(0, 1) == -1.0
        assert outcome_value_for_player(1, 1) == 1.0
        assert outcome_value_for_player(1, 0) == -1.0
        assert outcome_value_for_player(None, 0) == 0.0
        assert outcome_value_for_player(None, 1) == 0.0

    def test_is_zero_sum_for_two_players(self) -> None:
        for winner in (0, 1, None):
            assert outcome_value_for_player(winner, 0) == -outcome_value_for_player(winner, 1)

    def test_generalizes_beyond_two_players(self) -> None:
        assert outcome_value_for_player(2, 2) == 1.0
        assert [outcome_value_for_player(2, p) for p in range(4)] == [-1.0, -1.0, 1.0, -1.0]


class TestSequentialSelfPlayValueSigns:
    @pytest.mark.timeout(300)
    def test_targets_are_signed_per_player(self, small_model) -> None:
        from alpha_zero.cloud.gcp.selfplay_worker import run_selfplay

        trajectories = run_selfplay(
            model=small_model,
            num_games=2,
            mcts_sims=_MCTS_SIMS,
            mcts_type="gumbel",
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )
        assert trajectories, "self-play produced no trajectories"
        for trajectory in trajectories:
            assert "players" in trajectory, "trajectory must record the acting player"
            assert_value_targets_are_zero_sum(trajectory)

    @pytest.mark.timeout(300)
    def test_both_seats_are_represented(self, small_model) -> None:
        """Both players must actually appear, or the invariant is vacuous."""
        from alpha_zero.cloud.gcp.selfplay_worker import run_selfplay

        trajectories = run_selfplay(
            model=small_model,
            num_games=2,
            mcts_sims=_MCTS_SIMS,
            mcts_type="gumbel",
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )
        assert trajectories
        assert any(len(set(t["players"])) > 1 for t in trajectories), (
            "no trajectory contained moves from more than one seat"
        )


class TestBatchSelfPlayValueSigns:
    @pytest.mark.timeout(300)
    def test_targets_are_signed_per_player(self, small_model) -> None:
        from alpha_zero.cloud.gcp.selfplay_worker import run_batch_selfplay

        trajectories = run_batch_selfplay(
            model=small_model,
            num_games=2,
            mcts_sims=_MCTS_SIMS,
            board_size=_BOARD_SIZE,
            max_turns=_MAX_TURNS,
            num_parallel_games=2,
            max_batch_size=4,
            gumbel_num_actions=_GUMBEL_ACTIONS,
            device="cpu",
        )
        assert trajectories, "batch self-play produced no trajectories"
        for trajectory in trajectories:
            assert "players" in trajectory, "trajectory must record the acting player"
            assert_value_targets_are_zero_sum(trajectory)

    @pytest.mark.timeout(300)
    def test_batch_and_sequential_use_the_same_label_alphabet(self, small_model) -> None:
        """C3: the two paths previously disagreed on unfinished games.

        Batch labelled every position of a still-active game ``0.0`` while the
        sequential path used the last observed reward, so the same position could
        be labelled differently depending only on which worker produced it.
        """
        from alpha_zero.cloud.gcp.selfplay_worker import run_batch_selfplay, run_selfplay

        common = {
            "model": small_model,
            "num_games": 2,
            "mcts_sims": _MCTS_SIMS,
            "board_size": _BOARD_SIZE,
            "max_turns": _MAX_TURNS,
            "gumbel_num_actions": _GUMBEL_ACTIONS,
            "device": "cpu",
        }
        seq = run_selfplay(mcts_type="gumbel", **common)
        batch = run_batch_selfplay(num_parallel_games=2, max_batch_size=4, **common)

        allowed = {-1.0, 0.0, 1.0}
        for trajectories, label in ((seq, "sequential"), (batch, "batch")):
            assert trajectories, f"{label} produced no trajectories"
            observed = {float(r) for t in trajectories for r in t["rewards"]}
            assert observed <= allowed, f"{label} emitted values outside {allowed}: {observed}"
