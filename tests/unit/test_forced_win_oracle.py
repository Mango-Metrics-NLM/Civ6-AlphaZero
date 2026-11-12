"""Forced-win oracle over ``batch_gumbel`` and ``muzero`` (T18, LF-C4).

Why this test exists
--------------------
``tests/unit/test_mcts_backup_correctness.py`` holds the forced-win oracle for
``gumbel``, ``puct`` and ``simple``: a synthetic environment in which one player
moves ``depth`` times in a row and the first move decides the game. It is the
only gate that can see a sign defect, because a parity-scrambled value still
produces plausible aggregate win rates (CLAUDE.md 2026-09-04).

The two remaining searches were outside it:

* ``BatchGumbelMCTS`` is the batched self-play path. It ranked its sequential
  halving survivors with an unconditional ``q = -child.value`` -- the plain
  negamax rule the 2026-09-04 fix removed from every other variant. Across a
  same-player edge that ranks the winning move *last*.
* ``MuZeroMCTS`` searches a learned latent model, so it cannot be handed the
  synthetic environment directly. :class:`ForcedWinDynamics` below is a
  deterministic stand-in "network" whose latent state encodes the environment
  exactly -- plies played, the first action, whether the game is over -- and
  whose dynamics emit the terminal outcome as the reward on the final
  transition, the convention ``MuZeroTrainer`` learns (absorbing states carry
  zero reward and zero value).

The fixture and its constants are imported from the original oracle so the two
files cannot drift apart: a change to what "forced win" means changes both.

Both searches are also run on :class:`AlternatingTurnEnv`, where every edge is a
player-change edge. ``batch_gumbel`` must pass it at every depth. ``muzero``
passes it only at odd depth and is ``xfail(strict=True)`` at even depth: its
latent search has no mover to sign a reward by (see ``_MUZERO_MOVER_REASON``).

Negative control
----------------
Making ``perspective_sign`` in ``core/mcts/backup.py`` always return ``-1.0``
(plain negamax) fails ``batch_gumbel`` here and ``gumbel`` in the original
oracle. Recorded in the T18 report.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
import torch

from alpha_zero.core.batch_mcts import BatchGumbelMCTS, BatchMCTSConfig
from alpha_zero.core.mcts.gumbel import GumbelMCTS, GumbelMCTSConfig
from alpha_zero.core.mcts.gumbel_ops import VALID_POLICY_TARGETS
from alpha_zero.core.mcts.muzero import MuZeroMCTS, MuZeroMCTSConfig
from tests.unit.test_mcts_backup_correctness import (
    FORCED_WIN_DEPTHS,
    NUM_ACTIONS,
    SEARCH_SEED,
    SEARCH_SIMULATIONS,
    WINNING_ACTIONS,
    ConsecutiveTurnEnv,
    _UninformativeNetwork,
)

# Latent layout for ForcedWinDynamics: one row, three features.
_PLIES, _FIRST_ACTION, _DONE = 0, 1, 2
_LATENT_WIDTH = 3
_NO_ACTION_YET = -1.0


class AlternatingTurnEnv(ConsecutiveTurnEnv):
    """``ConsecutiveTurnEnv`` with the mover changing on every ply.

    The outcome is unchanged -- the first action still decides it, and
    ``winner`` is still absolute -- so the right answer is the same. Only the
    edge type changes: every edge is a player-change edge, which is the case
    plain negamax already handled. A search that passes the same-player oracle
    but fails this one has over-corrected.
    """

    def clone(self) -> AlternatingTurnEnv:
        twin = AlternatingTurnEnv(self.depth, self.action_space.n, self.winning_action)
        twin.current_player = self.current_player
        twin._plies = self._plies
        twin._first_action = self._first_action
        return twin

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._first_action is None:
            self._first_action = int(action)
        self._plies += 1
        self.current_player = 1 - self.current_player
        return self.get_observation(), 0.0, self.done, False, self.get_info()


ENV_TYPES: dict[str, type[ConsecutiveTurnEnv]] = {
    "same_player": ConsecutiveTurnEnv,
    "alternating": AlternatingTurnEnv,
}


def _assert_prefers_the_win(
    policy: np.ndarray, *, search_name: str, depth: int, winning_action: int
) -> None:
    """The three oracle assertions, shared by every parametrisation here.

    Mirrors ``test_search_prefers_the_winning_action``: a policy that is
    uniform over two or more actions carries no answer, so its ``argmax`` would
    be decided by index order rather than by the search.
    """
    considered = policy[policy > 0]
    assert considered.size > 0, f"{search_name} returned an all-zero policy"
    assert not (considered.size > 1 and np.allclose(considered, considered[0])), (
        f"{search_name} at depth {depth} returned a policy that is uniform over "
        f"{considered.size} actions ({policy.round(3)}); its argmax is a tie-break."
    )
    chosen = int(policy.argmax())
    assert chosen == winning_action, (
        f"{search_name} at depth {depth} chose action {chosen}, but action "
        f"{winning_action} is a forced win for the player to move "
        f"(policy {policy.round(3)})."
    )


# --- batch_gumbel -------------------------------------------------------------


def _build_batch(num_simulations: int = SEARCH_SIMULATIONS) -> BatchGumbelMCTS:
    return BatchGumbelMCTS(
        _UninformativeNetwork(),
        BatchMCTSConfig(num_simulations=num_simulations, seed=SEARCH_SEED),
        device="cpu",
    )


@pytest.mark.parametrize("env_type", sorted(ENV_TYPES))
@pytest.mark.parametrize("depth", FORCED_WIN_DEPTHS)
@pytest.mark.parametrize("winning_action", WINNING_ACTIONS)
def test_batch_gumbel_prefers_the_winning_action(
    env_type: str, depth: int, winning_action: int
) -> None:
    """Single-environment batch search picks the forced win at every depth and index."""
    env_cls = ENV_TYPES[env_type]
    # Reset and read on the concrete class: the counter is a class attribute,
    # and a subclass's ``+=`` writes its own copy.
    env_cls.reset_terminal_counter()
    env = env_cls(depth=depth, winning_action=winning_action)

    policy, _value = _build_batch().search(env)

    assert env_cls.terminals_reached > 0, (
        f"batch_gumbel at depth {depth} never reached a terminal; the oracle would be vacuous"
    )
    _assert_prefers_the_win(
        np.asarray(policy),
        search_name=f"batch_gumbel[{env_type}]",
        depth=depth,
        winning_action=winning_action,
    )


@pytest.mark.parametrize("env_type", sorted(ENV_TYPES))
def test_batch_gumbel_keeps_each_tree_independent(env_type: str) -> None:
    """One ``batch_search`` over every (depth, index) case answers each one correctly.

    Batch keeps one tree per environment; this proves each tree's ``to_play``
    and ranking are its own -- a position is not answered by its neighbour's
    evidence -- and that mixed depths (different halving progress per tree)
    still resolve correctly.
    """
    env_cls = ENV_TYPES[env_type]
    env_cls.reset_terminal_counter()
    cases = [(d, w) for d in FORCED_WIN_DEPTHS for w in WINNING_ACTIONS]
    envs = [env_cls(depth=d, winning_action=w) for d, w in cases]

    results = _build_batch().batch_search(envs)

    assert len(results) == len(cases)
    assert env_cls.terminals_reached >= len(cases)
    for (depth, winning_action), (policy, _value) in zip(cases, results, strict=True):
        _assert_prefers_the_win(
            np.asarray(policy),
            search_name=f"batch_gumbel[{env_type}, batched]",
            depth=depth,
            winning_action=winning_action,
        )


@pytest.mark.parametrize("policy_target", sorted(VALID_POLICY_TARGETS))
@pytest.mark.parametrize("env_type", sorted(ENV_TYPES))
@pytest.mark.parametrize("depth", FORCED_WIN_DEPTHS)
@pytest.mark.parametrize("winning_action", WINNING_ACTIONS)
def test_batch_gumbel_single_env_is_gumbel(
    policy_target: str, env_type: str, depth: int, winning_action: int
) -> None:
    """Searched alone, a batch tree is exactly the ``GumbelMCTS`` tree.

    The completed-Q target re-signs each root Q itself, so the oracle above
    cannot see the halving *ranking*: reverting batch's survivor ranking to the
    unconditional ``-child.value`` still passes it (measured, T18). This test
    can. ``one_hot`` reports the halving winner directly, and lockstep stepping
    gives each batch tree GumbelMCTS's exact simulation sequence, so any
    divergence in sign, ranking, schedule or target is a mismatch.
    """
    env = ENV_TYPES[env_type](depth=depth, winning_action=winning_action)
    gumbel_policy, gumbel_value = GumbelMCTS(
        _UninformativeNetwork(),
        GumbelMCTSConfig(
            num_simulations=SEARCH_SIMULATIONS, seed=SEARCH_SEED, policy_target=policy_target
        ),
        device="cpu",
    ).search(env)
    batch_policy, batch_value = BatchGumbelMCTS(
        _UninformativeNetwork(),
        BatchMCTSConfig(
            num_simulations=SEARCH_SIMULATIONS, seed=SEARCH_SEED, policy_target=policy_target
        ),
        device="cpu",
    ).search(env)

    np.testing.assert_allclose(batch_policy, gumbel_policy)
    assert batch_value == pytest.approx(gumbel_value)


def test_batch_gumbel_records_to_play_on_every_node() -> None:
    """Every node the batch search creates carries its own environment's mover.

    The ranking fix relies on ``to_play`` being set below the root; a node left
    at ``None`` silently falls back to alternation for that edge.
    """
    search = _build_batch(num_simulations=32)
    envs = [ConsecutiveTurnEnv(depth=3), AlternatingTurnEnv(depth=3)]
    captured: list[list[Any]] = []
    original = search._backpropagate

    def spy(path: list[Any], value: float) -> None:
        captured.append(list(path))
        original(path, value)

    search._backpropagate = spy  # type: ignore[method-assign]
    search.batch_search(envs)

    assert captured, "no backup ran"
    seen_players: set[int] = set()
    for path in captured:
        for node in path:
            assert node.to_play is not None, "a node on a backed-up path has no to_play"
            seen_players.add(int(node.to_play))
    # The same-player env only ever has player 0; the alternating env reaches
    # player 1 at odd plies. Both must appear, or one tree borrowed the other's.
    assert seen_players == {0, 1}


# --- muzero -------------------------------------------------------------------


class ForcedWinDynamics(torch.nn.Module):
    """A deterministic MuZero "network" that encodes ``ConsecutiveTurnEnv`` exactly.

    * Latent state ``(1, 3)``: plies played, the first action (``-1`` before
      any), and a done flag.
    * Dynamics: one ply per step; the transition that ends the game carries the
      outcome as its reward -- ``+1`` when the first action was the winning one,
      ``-1`` otherwise -- from the perspective of the player who moved, which in
      the same-player game is always the root player. Every transition out of a
      finished state is absorbing: same latent, zero reward.
    * Prediction: uniform policy, zero value everywhere. As in the original
      oracle, the terminal outcome is the only signal.

    ``terminal_transitions`` is the vacuity guard: it counts transitions that
    actually produced the outcome.
    """

    terminal_transitions: ClassVar[int] = 0

    def __init__(self, depth: int, winning_action: int, *, alternating: bool = False) -> None:
        super().__init__()
        self.depth = depth
        self.winning_action = winning_action
        self.alternating = alternating

    @classmethod
    def reset_counter(cls) -> None:
        cls.terminal_transitions = 0

    def initial_inference(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        hidden = torch.tensor([[0.0, _NO_ACTION_YET, 0.0]])
        return SimpleNamespace(
            hidden_state=hidden,
            policy_logits=torch.zeros(1, NUM_ACTIONS),
            value=torch.zeros(1, 1),
        )

    def recurrent_inference(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        plies, first, done = (float(v) for v in hidden_state[0, :_LATENT_WIDTH])
        reward = 0.0
        if done < 0.5:
            if first == _NO_ACTION_YET:
                first = float(int(action[0]))
            plies += 1.0
            if plies >= self.depth:
                done = 1.0
                type(self).terminal_transitions += 1
                root_wins = int(first) == self.winning_action
                # Reward from the mover's perspective. In the alternating game
                # the final mover is the root player only at odd depth.
                mover_is_root = (not self.alternating) or (int(plies) % 2 == 1)
                reward = 1.0 if root_wins == mover_is_root else -1.0
        next_hidden = torch.tensor([[plies, first, done]])
        return (
            next_hidden,
            torch.tensor([[reward]]),
            torch.zeros(1, NUM_ACTIONS),
            torch.zeros(1, 1),
        )

    def prediction_function(
        self,
        hidden_state: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1, NUM_ACTIONS), torch.zeros(1, 1)


def _search_muzero(env: ConsecutiveTurnEnv, *, alternating: bool) -> np.ndarray:
    network = ForcedWinDynamics(env.depth, env.winning_action, alternating=alternating)
    # The deployed configuration (MuZeroTrainingBridge builds the default:
    # Gumbel root, sequential halving), with a scalar reward head because the
    # stand-in network emits scalars.
    search = MuZeroMCTS(
        network,
        MuZeroMCTSConfig(
            num_simulations=SEARCH_SIMULATIONS,
            seed=SEARCH_SEED,
            use_categorical_reward=False,
        ),
        device="cpu",
    )
    policy, _value = search.search_from_env(env, add_noise=False)
    return np.asarray(policy)


@pytest.mark.parametrize("depth", FORCED_WIN_DEPTHS)
@pytest.mark.parametrize("winning_action", WINNING_ACTIONS)
def test_muzero_prefers_the_winning_action(depth: int, winning_action: int) -> None:
    """MuZero's default search picks the forced win in the same-player game."""
    ForcedWinDynamics.reset_counter()
    env = ConsecutiveTurnEnv(depth=depth, winning_action=winning_action)

    policy = _search_muzero(env, alternating=False)

    assert ForcedWinDynamics.terminal_transitions > 0, (
        f"muzero at depth {depth} never produced a terminal transition; vacuous"
    )
    _assert_prefers_the_win(
        policy, search_name="muzero[same_player]", depth=depth, winning_action=winning_action
    )


#: Why MuZero's alternating case fails at even depth. Its Gumbel path backs up
#: single-agent discounted returns (``_backpropagate_with_rewards``, the
#: convention ``MuZeroTrainer`` learns) with no sign flip anywhere, because a
#: latent model has no ``to_play`` to flip on. A reward emitted from the
#: *mover's* perspective therefore reaches the root with the right sign only
#: when the final mover is the root player -- odd depth here. At even depth the
#: opponent made the last move, so a win for the root arrives as ``-1``. That is
#: a limitation of searching a latent model without a predicted mover, not a
#: defect this oracle can be fixed to accept; ``strict=True`` turns the xfail
#: into a failure the day a mover-aware MuZero makes it pass.
_MUZERO_MOVER_REASON = "latent model cannot observe the mover"


@pytest.mark.parametrize(
    "depth",
    [
        pytest.param(
            depth,
            marks=pytest.mark.xfail(strict=True, reason=_MUZERO_MOVER_REASON),
        )
        if depth % 2 == 0
        else depth
        for depth in FORCED_WIN_DEPTHS
    ],
)
@pytest.mark.parametrize("winning_action", WINNING_ACTIONS)
def test_muzero_prefers_the_winning_action_alternating(depth: int, winning_action: int) -> None:
    """In the alternating game MuZero finds the win only when the root moves last."""
    ForcedWinDynamics.reset_counter()
    env = AlternatingTurnEnv(depth=depth, winning_action=winning_action)

    policy = _search_muzero(env, alternating=True)

    assert ForcedWinDynamics.terminal_transitions > 0
    _assert_prefers_the_win(
        policy, search_name="muzero[alternating]", depth=depth, winning_action=winning_action
    )


def test_fixture_dynamics_match_the_environment() -> None:
    """Guard the stand-in: its rewards must agree with the real environment's winner."""
    for env_type, env_cls in ENV_TYPES.items():
        for depth in FORCED_WIN_DEPTHS:
            for winning_action in WINNING_ACTIONS:
                for first in range(NUM_ACTIONS):
                    _check_dynamics(
                        env_cls, env_type == "alternating", depth, winning_action, first
                    )


def _check_dynamics(
    env_cls: type[ConsecutiveTurnEnv],
    alternating: bool,
    depth: int,
    winning_action: int,
    first: int,
) -> None:
    net = ForcedWinDynamics(depth, winning_action, alternating=alternating)
    hidden = net.initial_inference(torch.zeros(1), torch.zeros(1)).hidden_state
    env = env_cls(depth=depth, winning_action=winning_action)
    rewards = []
    last_mover = env.current_player
    for _ in range(depth):
        hidden, reward, _, _ = net.recurrent_inference(hidden, torch.tensor([first]))
        rewards.append(float(reward))
        last_mover = env.current_player
        env.step(first)
    assert env.done
    # The reward is from the perspective of the player who made the final move.
    expected = 1.0 if env.get_info()["winner"] == last_mover else -1.0
    assert rewards[:-1] == [0.0] * (depth - 1)
    assert rewards[-1] == expected
    # Absorbing afterwards.
    _, reward, _, _ = net.recurrent_inference(hidden, torch.tensor([first]))
    assert float(reward) == 0.0
