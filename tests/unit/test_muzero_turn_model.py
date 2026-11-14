"""MuZero's mover model, signed reward backup and training-target frames (2026-11-12).

Why this test exists
--------------------
``MuZeroMCTS`` searches a learned latent model, so it cannot ask an environment
who is to move. Until 2026-11-12 it did not track the mover at all: its backup
summed rewards with no sign flip, so on two-player MicroCiv an opponent's gain
counted as the root player's (``tests/unit/test_forced_win_oracle.py`` failed
every even-depth alternating case). The fix leans on three facts this module
pins against the real environment, because the oracle runs on a toy game and
cannot see them:

1. ``MicroCivEnv`` changes ``current_player`` only on END_TURN, so
   :class:`EndTurnModel` predicts the mover exactly -- checked step by step
   over seeded random games, not assumed from reading ``_end_turn``.
2. The env scores the outcome against the player to move *after* the step,
   which is the frame :mod:`alpha_zero.core.mcts.muzero_tree` gives an edge's
   reward.
3. Both self-play paths now write targets in those frames through one helper,
   :mod:`alpha_zero.training.muzero_targets`.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from alpha_zero.core.mcts.backup import perspective_sign
from alpha_zero.core.mcts.muzero import MuZeroMCTS, MuZeroMCTSConfig
from alpha_zero.core.mcts.muzero_tree import (
    MICROCIV_END_TURN_ACTION,
    EndTurnModel,
    MinMaxStats,
    MuZeroNode,
    backpropagate_with_rewards,
    edge_q,
)
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.training.muzero_targets import outcome_for, outcome_only_targets
from alpha_zero.utils.constants import REWARDS

_SEEDS = (0, 1, 2, 3, 4, 5)
_BOARD = 8
_MAX_TURNS = 6


def _random_game(seed: int) -> list[tuple[int, int, int, float, bool, dict[str, Any]]]:
    """``(player_before, action, player_after, reward, done, info)`` per step."""
    env = MicroCivEnv(board_size=_BOARD, max_turns=_MAX_TURNS, seed=seed)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    steps = []
    done = False
    while not done:
        before = int(env.current_player)
        action = int(rng.choice(np.flatnonzero(env.get_valid_actions())))
        _obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps.append((before, action, int(env.current_player), float(reward), done, info))
    return steps


# --- 1. the mover model agrees with the environment -------------------------


def test_end_turn_constant_is_the_action_spaces() -> None:
    env = MicroCivEnv(board_size=_BOARD, max_turns=_MAX_TURNS, seed=0)
    assert env.action_space_handler.end_turn_offset == MICROCIV_END_TURN_ACTION
    assert MuZeroMCTSConfig().end_turn_action == MICROCIV_END_TURN_ACTION


@pytest.mark.parametrize("seed", _SEEDS)
def test_end_turn_model_predicts_every_mover_change(seed: int) -> None:
    model = EndTurnModel()
    steps = _random_game(seed)
    changes = 0
    for before, action, after, _reward, _done, _info in steps:
        assert model(before, action) == after, (
            f"seed {seed}: action {action} by player {before} left player {after} "
            f"to move, but the turn model predicted {model(before, action)}"
        )
        changes += before != after
    # Non-vacuity: the game crossed several turns, so both edge types occurred.
    assert changes >= 2
    assert changes < len(steps)


def test_terminal_reward_is_in_the_next_movers_frame() -> None:
    """The env scores the outcome for the player to move after the final step.

    A turn-limit end only follows END_TURN, which earns no shaped reward, so
    the final reward is exactly the outcome in that player's frame -- the
    opposite of the frame of the player who ended the game.
    """
    decisive = 0
    for seed in _SEEDS:
        before, action, after, reward, done, info = _random_game(seed)[-1]
        assert done
        # These short random games all end at the turn limit (measured).
        assert action == MICROCIV_END_TURN_ACTION
        assert before != after
        assert reward == outcome_for(info["winner"], after) * REWARDS.WIN_REWARD
        decisive += info["winner"] is not None
    # Non-vacuity: a draw scores 0 in every frame and so proves nothing.
    assert decisive >= 1


# --- 2. the signed, reward-aware backup ---------------------------------------


def _path(players: list[int], rewards: list[float]) -> list[MuZeroNode]:
    nodes = [MuZeroNode(to_play=players[0])]
    for player, reward in zip(players[1:], rewards, strict=True):
        nodes.append(MuZeroNode(to_play=player, reward=reward, parent=nodes[-1]))
    return nodes


def test_backup_signs_each_edge_by_mover_change() -> None:
    """Hand-computed: root(p0) -same-> a(p0) -change-> b(p1) -same-> leaf(p1)."""
    discount = 0.5
    path = _path([0, 0, 1, 1], rewards=[0.1, 0.2, 0.4])
    backpropagate_with_rewards(path, leaf_value=0.8, discount=discount)

    leaf, b, a, root = path[3].value, path[2].value, path[1].value, path[0].value
    assert leaf == pytest.approx(0.8)
    assert b == pytest.approx(0.4 + discount * 0.8)  # same player: no flip
    assert a == pytest.approx(-(0.2 + discount * b))  # player change: flip
    assert root == pytest.approx(0.1 + discount * a)
    assert all(node.visit_count == 1 for node in path)


def test_edge_q_matches_the_backup_and_updates_bounds() -> None:
    stats = MinMaxStats()
    path = _path([0, 1], rewards=[1.0])
    backpropagate_with_rewards(path, leaf_value=0.0, discount=1.0, stats=stats)
    # Player 1 gained the reward on the edge into its node: bad for player 0.
    assert edge_q(path[0], path[1], 1.0) == pytest.approx(-1.0)
    assert path[0].value == pytest.approx(-1.0)
    assert stats.minimum == stats.maximum == pytest.approx(-1.0)


def test_min_max_stats_normalise_and_pass_through_until_spread() -> None:
    stats = MinMaxStats()
    assert stats.normalize(0.3) == 0.3
    stats.update(-1.0)
    stats.update(1.0)
    assert stats.normalize(0.0) == pytest.approx(0.5)
    assert stats.normalize(1.0) == pytest.approx(1.0)


def test_equal_q_without_spread_does_not_favour_visited_children() -> None:
    """Copilot review on PR #136: a 0.5 no-spread fallback scored a visited child
    above an unvisited one at equal Q, so selection never left the visited branch.
    """
    parent = MuZeroNode(to_play=0, visit_count=1)
    parent.children[0] = MuZeroNode(to_play=0, prior=0.5, visit_count=1, value_sum=0.0)
    parent.children[1] = MuZeroNode(to_play=0, prior=0.5)
    stats = MinMaxStats()
    stats.update(0.0)  # one Q seen: no spread
    action, _child = parent.select_child(1.5, stats.minimum, stats.maximum)
    assert action == 1, "at equal Q the unvisited child's exploration term must win"


def test_selection_prefers_the_edge_that_is_good_for_the_parent() -> None:
    """A player-change child valued +1 by its own mover is a loss for the parent."""
    parent = MuZeroNode(to_play=0, visit_count=2)
    parent.children[0] = MuZeroNode(to_play=1, prior=0.5, visit_count=1, value_sum=1.0)
    parent.children[1] = MuZeroNode(to_play=0, prior=0.5, visit_count=1, value_sum=0.2)
    assert perspective_sign(0, 1) == -1.0
    action, _child = parent.select_child(c_puct=0.0, min_q=-1.0, max_q=1.0)
    assert action == 1


def test_expand_assigns_each_child_its_mover() -> None:
    node = MuZeroNode(to_play=1)
    node.expand([0, 5, 7], np.full(3, 1 / 3), EndTurnModel())
    assert {a: c.to_play for a, c in node.children.items()} == {0: 0, 5: 1, 7: 1}


# --- 3. the search wires it through -------------------------------------------


class _TinyMuZero(torch.nn.Module):
    """Constant model: uniform policy, value 0.25, reward 0.5 on every edge."""

    def __init__(self, actions: int) -> None:
        super().__init__()
        self.actions = actions

    def initial_inference(self, spatial: Any, vector: Any, mask: Any = None) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            hidden_state=torch.zeros(1, 1),
            policy_logits=torch.zeros(1, self.actions),
            value=torch.full((1, 1), 0.25),
        )

    def recurrent_inference(self, hidden: Any, action: Any, mask: Any = None) -> Any:
        return hidden, torch.full((1, 1), 0.5), torch.zeros(1, self.actions), torch.zeros(1, 1)

    def prediction_function(self, hidden: Any, mask: Any = None) -> Any:
        return torch.zeros(1, self.actions), torch.full((1, 1), 0.25)


@pytest.mark.parametrize("use_gumbel", [True, False])
@pytest.mark.parametrize("root_player", [0, 1])
def test_every_backed_up_node_carries_the_turn_models_mover(
    use_gumbel: bool, root_player: int
) -> None:
    actions = 4
    search = MuZeroMCTS(
        _TinyMuZero(actions),
        MuZeroMCTSConfig(
            num_simulations=24,
            num_sampled_actions=4,
            seed=0,
            use_categorical_reward=False,
            use_gumbel=use_gumbel,
        ),
    )
    paths: list[list[MuZeroNode]] = []
    original = search._backpropagate_with_rewards

    def spy(path: list[MuZeroNode], value: float) -> None:
        paths.append(list(path))
        original(path, value)

    search._backpropagate_with_rewards = spy  # type: ignore[method-assign]
    observation = {"spatial": np.zeros((1, 2, 2), np.float32), "vector": np.zeros(2, np.float32)}
    search.search(observation, np.ones(actions, np.float32), add_noise=False, to_play=root_player)

    assert paths
    assert {p[0].to_play for p in paths} == {root_player}
    crossed = 0
    for path in paths:
        for parent, child in itertools.pairwise(path):
            assert child.to_play == EndTurnModel()(parent.to_play, child.action)
            crossed += parent.to_play != child.to_play
    assert crossed > 0, "no END_TURN edge was searched; the check is vacuous"


def test_search_from_env_roots_at_the_envs_mover() -> None:
    env = MicroCivEnv(board_size=_BOARD, max_turns=_MAX_TURNS, seed=0)
    env.reset(seed=0)
    env.step(MICROCIV_END_TURN_ACTION)
    assert env.current_player == 1
    search = MuZeroMCTS(
        _TinyMuZero(env.action_space.n),
        MuZeroMCTSConfig(num_simulations=8, seed=0, use_categorical_reward=False),
    )
    roots: list[MuZeroNode] = []
    original = search._gumbel_policy_target

    def spy(root: MuZeroNode, best: int, size: int) -> np.ndarray:
        roots.append(root)
        return original(root, best, size)

    search._gumbel_policy_target = spy  # type: ignore[method-assign]
    search.search_from_env(env, add_noise=False)
    assert roots[0].to_play == 1


def test_from_network_keeps_the_turn_rule() -> None:
    def never(to_play: int, action: int) -> int:
        return to_play

    search = MuZeroMCTS.from_network(
        _TinyMuZero(3), MuZeroMCTSConfig(end_turn_action=2), turn_model=never
    )
    assert search.config.end_turn_action == 2
    assert search._turn_model is never
    assert MuZeroMCTS(_TinyMuZero(3), MuZeroMCTSConfig(end_turn_action=2))._turn_model == (
        EndTurnModel(2)
    )


def test_end_turn_model_refuses_a_third_player() -> None:
    with pytest.raises(ValueError, match="two-player"):
        EndTurnModel()(2, 0)


def test_negative_end_turn_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="end_turn_action"):
        MuZeroMCTSConfig(end_turn_action=-1)


# --- 4. training targets ------------------------------------------------------


def test_outcome_only_targets() -> None:
    rewards, values = outcome_only_targets(winner=1, players=[0, 0, 1, 1, 0], final_to_play=1)
    assert values == [-1.0, -1.0, 1.0, 1.0, -1.0]
    assert rewards == [0.0, 0.0, 0.0, 0.0, 1.0]
    assert outcome_only_targets(None, [0, 1], 0) == ([0.0, 0.0], [0.0, 0.0])
    assert outcome_only_targets(0, [], 1) == ([], [])


def test_outcome_rule_matches_the_cloud_workers() -> None:
    from alpha_zero.cloud.gcp.selfplay_worker import outcome_value_for_player

    for winner in (None, 0, 1):
        for player in (0, 1):
            assert outcome_for(winner, player) == outcome_value_for_player(winner, player)


class _DecisiveTargetsSpy:
    """Wraps ``outcome_only_targets``: records what a self-play path passes, forces a winner.

    The frame checks below are vacuous on a draw -- every target is 0 in every
    frame -- and short random games are mostly draws. So a draw is replaced by
    a win for the first mover before the real helper runs; the path must still
    store exactly what the helper returned, in the right fields.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int | None, list[int], int]] = []
        self.returned: tuple[list[float], list[float]] | None = None

    def __call__(
        self, winner: int | None, players: Any, final_to_play: int
    ) -> tuple[list[float], list[float]]:
        players = [int(p) for p in players]
        self.calls.append((winner, players, int(final_to_play)))
        forced = players[0] if winner is None else winner
        rewards, values = outcome_only_targets(forced, players, final_to_play)
        self.returned = (list(rewards), list(values))
        return rewards, values


def _assert_mover_frame(
    spy: _DecisiveTargetsSpy,
    players: list[int],
    actions: list[int],
    values: list[float],
    rewards: list[float],
) -> None:
    """The path passed the right movers and stored the decisive targets verbatim."""
    assert len(spy.calls) == 1, "the targets must be built once per game"
    _winner, passed_players, final_to_play = spy.calls[0]
    assert passed_players == players, "players must be the movers BEFORE each step"
    assert final_to_play == EndTurnModel()(players[-1], actions[-1]), (
        "final_to_play must be the player to move AFTER the last step"
    )
    assert spy.returned == (rewards, values)
    assert set(players) == {0, 1}, "the game must have crossed a turn to test the frame"
    # Non-vacuity: decisive, so a sign error in either frame changes a value.
    assert values[0] != 0.0
    for player, value in zip(players, values, strict=True):
        assert value == (values[0] if player == players[0] else -values[0])
    assert rewards[:-1] == [0.0] * (len(rewards) - 1)
    assert rewards[-1] == values[players.index(final_to_play)]


def test_run_muzero_training_targets_are_in_the_mover_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import run_muzero_training
    from run_muzero_training import (
        MuZeroRunConfig,
        create_mcts,
        create_network,
        self_play_game,
    )

    spy = _DecisiveTargetsSpy()
    monkeypatch.setattr(run_muzero_training, "outcome_only_targets", spy)
    config = MuZeroRunConfig(
        board_size=_BOARD,
        max_turns=3,
        num_simulations=4,
        num_res_blocks=1,
        num_filters=16,
        device="cpu",
    )
    env = MicroCivEnv(board_size=_BOARD, max_turns=3, seed=0)
    torch.manual_seed(0)
    mcts = create_mcts(create_network(config, env.action_space.n), config)

    players: list[int] = []
    step = env.step

    def recording_step(action: int) -> Any:
        players.append(int(env.current_player))
        return step(action)

    env.step = recording_step  # type: ignore[method-assign]
    trajectory = self_play_game(env, mcts, config)

    assert spy.calls[0][0] == env.winner
    assert spy.calls[0][2] == env.current_player
    values = [float(v) for v in trajectory.values]
    assert values[-1] == 0.0, "the absorbing state after the game is worth 0"
    _assert_mover_frame(
        spy,
        players,
        [int(a) for a in trajectory.actions],
        values[:-1],
        [float(r) for r in trajectory.rewards],
    )


def test_cloud_muzero_selfplay_targets_are_in_the_mover_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import alpha_zero.training.muzero_targets as targets_module
    from alpha_zero.cloud.gcp.selfplay_worker import run_muzero_selfplay
    from alpha_zero.core.muzero_network import MuZeroConfig, MuZeroNetwork

    # run_muzero_selfplay imports the helper inside the function, so patching
    # the module attribute reaches it.
    spy = _DecisiveTargetsSpy()
    monkeypatch.setattr(targets_module, "outcome_only_targets", spy)
    env = MicroCivEnv(board_size=_BOARD, max_turns=3, seed=0)
    torch.manual_seed(0)
    model = MuZeroNetwork(
        MuZeroConfig(
            spatial_channels=13,
            spatial_size=_BOARD,
            vector_dim=64,
            num_res_blocks=1,
            num_filters=16,
            action_space_size=env.action_space.n,
        )
    )
    trajectories = run_muzero_selfplay(
        model, num_games=1, mcts_sims=4, board_size=_BOARD, max_turns=3
    )
    assert len(trajectories) == 1
    traj = trajectories[0]
    _assert_mover_frame(
        spy,
        [int(p) for p in traj["players"]],
        [int(a) for a in traj["actions"]],
        [float(v) for v in traj["rewards"]],
        [float(r) for r in traj["reward_targets"]],
    )
