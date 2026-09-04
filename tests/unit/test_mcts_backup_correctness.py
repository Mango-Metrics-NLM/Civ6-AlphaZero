"""Forced-win oracle: does a search prefer the action that actually wins?

Why this test exists
--------------------
Every MCTS variant in this package stored a node's value from the acting
player's perspective but flipped the sign on *every* tree edge (plain negamax).
That is correct only in a game with exactly one ply per turn. MicroCiv advances
``current_player`` only inside ``_end_turn``, so a player moves each unit before
yielding and most edges are same-player edges. Under the old rule the sign of a
node's mean value depended on the *parity* of the depth at which its leaves were
evaluated, so a search could prefer a losing move.

Aggregate metrics hide this: a parity-scrambled value still produces
plausible-looking win rates. Only a position with a known right answer exposes
it, which is what ``ConsecutiveTurnEnv`` provides.

The environment here is deliberately not MicroCiv. It isolates the one property
that matters -- consecutive moves by one player -- so a failure means "the
backup sign is wrong", not "the game is hard".

Negative control (``.claude/skills/negative-control``): reverting
``backpropagate_path`` to an unconditional ``value = -value`` must fail
``test_search_prefers_the_winning_action`` at odd depths. That was confirmed
before the fix was written.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from alpha_zero.core.mcts.backup import backpropagate_path, backup_edge_summary

# Deliberately small: the oracle must be cheap enough to run on every CI job.
#
# WINNING_ACTION is parameterised, never fixed at 0. An earlier version of this
# file pinned it to 0 and the oracle passed vacuously for the Gumbel variant:
# the returned policy was exactly uniform ([1/3, 1/3, 1/3]) and `np.argmax`
# breaks a tie by returning the LOWEST index, which was the winning action by
# coincidence. Moving the win to index 1 or 2 failed at every depth. A tie-broken
# argmax is not evidence, so the winning index is now swept.
WINNING_ACTIONS = (0, 1, 2)
WINNING_ACTION = 0  # module-level default for fixtures that do not sweep
NUM_ACTIONS = 3
SPATIAL_SHAPE = (2, 3, 3)
VECTOR_SIZE = 4
# Depths span both parities. The old unconditional flip is correct at even depth
# and inverted at odd depth, so a single depth could pass by luck.
#
# Bounded at 4 on purpose. A search that reaches an unexpanded node backs up the
# network's value (zero, by construction here) instead of continuing to a
# terminal, so beyond the budget's reach the terminal signal never arrives and
# the choice becomes arbitrary. A flaky oracle is worse than no oracle, so the
# depths stay inside what SEARCH_SIMULATIONS can reach and
# `terminals_reached` proves it on every run.
FORCED_WIN_DEPTHS = (1, 2, 3, 4)
SEARCH_SIMULATIONS = 128
SEARCH_SEED = 20260904


class _ActionSpace:
    """Minimal action space exposing the ``n`` the protocol requires."""

    def __init__(self, n: int) -> None:
        self.n = n


class ConsecutiveTurnEnv:
    """A game where one player moves ``depth`` times in a row, then it ends.

    The first action decides the game: ``WINNING_ACTION`` makes player 0 the
    winner, anything else makes player 1 the winner. Every ply is played by the
    same player, so every tree edge is a same-player edge and no sign flip is
    ever correct.

    Satisfies :class:`alpha_zero.core.mcts.mcts_protocols.MCTSEnvironment`
    structurally; it is not registered as a gym env on purpose.
    """

    # Class-level so it survives clone(): the search simulates on copies, and a
    # per-instance counter would never be seen by the test. This is the vacuity
    # guard -- it proves a terminal was actually reached, so the oracle cannot
    # pass simply because no signal ever arrived.
    terminals_reached = 0

    def __init__(
        self,
        depth: int,
        num_actions: int = NUM_ACTIONS,
        winning_action: int = WINNING_ACTION,
    ) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1; got {depth}")
        self.depth = depth
        self.winning_action = winning_action
        self.action_space = _ActionSpace(num_actions)
        self.current_player = 0
        self._plies = 0
        self._first_action: int | None = None

    @classmethod
    def reset_terminal_counter(cls) -> None:
        cls.terminals_reached = 0

    def clone(self) -> ConsecutiveTurnEnv:
        twin = ConsecutiveTurnEnv(self.depth, self.action_space.n, self.winning_action)
        twin.current_player = self.current_player
        twin._plies = self._plies
        twin._first_action = self._first_action
        return twin

    @property
    def done(self) -> bool:
        return self._plies >= self.depth

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._first_action is None:
            self._first_action = int(action)
        self._plies += 1
        # current_player never changes: this is the whole point of the fixture.
        return self.get_observation(), 0.0, self.done, False, self.get_info()

    def get_observation(self) -> dict[str, np.ndarray]:
        return {
            "spatial": np.zeros(SPATIAL_SHAPE, dtype=np.float32),
            "vector": np.zeros(VECTOR_SIZE, dtype=np.float32),
            "action_mask": self.get_valid_actions(),
        }

    def get_valid_actions(self) -> np.ndarray:
        return np.ones(self.action_space.n, dtype=np.float32)

    def get_info(self) -> dict[str, Any]:
        if not self.done:
            return {}
        type(self).terminals_reached += 1
        winner = 0 if self._first_action == self.winning_action else 1
        return {"winner": winner}


class _UninformativeNetwork(torch.nn.Module):
    """Uniform policy, zero value: the terminal outcome is the only signal.

    If the network carried any signal the oracle could pass for the wrong
    reason, so it deliberately carries none.
    """

    def forward(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = spatial.shape[0] if spatial.dim() > 3 else 1
        logits = torch.zeros(batch, NUM_ACTIONS)
        value = torch.zeros(batch, 1)
        return logits, value

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = spatial.shape[0] if spatial.dim() > 3 else 1
        policy = torch.full((batch, NUM_ACTIONS), 1.0 / NUM_ACTIONS)
        value = torch.zeros(batch, 1)
        return policy, value


def _build_gumbel(num_simulations: int) -> Any:
    from alpha_zero.core.mcts.gumbel import GumbelMCTS, GumbelMCTSConfig

    return GumbelMCTS(
        _UninformativeNetwork(),
        GumbelMCTSConfig(num_simulations=num_simulations, seed=SEARCH_SEED),
        device="cpu",
    )


def _build_puct(num_simulations: int) -> Any:
    from alpha_zero.core.mcts.base import MCTSConfig
    from alpha_zero.core.mcts.puct import Civ6MCTS

    return Civ6MCTS(
        _UninformativeNetwork(),
        MCTSConfig(num_simulations=num_simulations, seed=SEARCH_SEED),
        device="cpu",
    )


def _build_simple(num_simulations: int) -> Any:
    # SimpleMCTS is the one variant that takes plain arguments, not a config.
    from alpha_zero.core.mcts.simple import SimpleMCTS

    return SimpleMCTS(num_simulations=num_simulations, seed=SEARCH_SEED)


# Registry so a new variant is one entry, not a new copy of the test body.
SEARCH_BUILDERS = {
    "gumbel": _build_gumbel,
    "puct": _build_puct,
    "simple": _build_simple,
}


@pytest.mark.parametrize("search_name", sorted(SEARCH_BUILDERS))
@pytest.mark.parametrize("depth", FORCED_WIN_DEPTHS)
@pytest.mark.parametrize("winning_action", WINNING_ACTIONS)
def test_search_prefers_the_winning_action(
    search_name: str, depth: int, winning_action: int
) -> None:
    """The search must pick the action that wins, at every depth and every index."""
    ConsecutiveTurnEnv.reset_terminal_counter()
    env = ConsecutiveTurnEnv(depth=depth, winning_action=winning_action)
    search = SEARCH_BUILDERS[search_name](num_simulations=SEARCH_SIMULATIONS)

    policy, _value = search.search(env)
    policy = np.asarray(policy)

    assert ConsecutiveTurnEnv.terminals_reached > 0, (
        f"{search_name} at depth {depth} never reached a terminal state, so the "
        "outcome signal never reached the root and this assertion would pass or "
        "fail at random. Raise SEARCH_SIMULATIONS or lower the depth."
    )

    # A uniform policy carries no answer. Without this, `argmax` silently returns
    # index 0 and the test passes whenever the win happens to sit there -- which
    # is exactly how this oracle was vacuous for the Gumbel variant.
    considered = policy[policy > 0]
    assert considered.size > 0, f"{search_name} returned an all-zero policy"
    # A single non-zero entry is maximal information, not a tie -- only a TIE
    # across two or more actions is uninformative.
    assert not (considered.size > 1 and np.allclose(considered, considered[0])), (
        f"{search_name} at depth {depth} returned a policy that is uniform over "
        f"{considered.size} actions ({policy.round(3)}). It contains no preference, "
        "so any assertion on its argmax is decided by tie-breaking, not by search."
    )

    chosen = int(policy.argmax())
    assert chosen == winning_action, (
        f"{search_name} at depth {depth} chose action {chosen}, "
        f"but action {winning_action} is a forced win for the player to move. "
        "A value backup that flips on same-player edges inverts the sign at odd depth."
    )


def test_fixture_has_only_same_player_edges() -> None:
    """Guard the fixture itself: if it alternated, the oracle would prove nothing."""
    env = ConsecutiveTurnEnv(depth=5)
    players = [env.current_player]
    while not env.done:
        env.step(WINNING_ACTION)
        players.append(env.current_player)
    assert len(set(players)) == 1, f"fixture must not change player, saw {players}"


class _Node:
    """Minimal structural node for testing the backup primitive directly."""

    def __init__(self, to_play: int | None) -> None:
        self.visit_count = 0
        self.value_sum = 0.0
        self.to_play = to_play


def test_backpropagate_path_does_not_flip_on_same_player_edges() -> None:
    path = [_Node(0), _Node(0), _Node(0)]
    backpropagate_path(path, 1.0)
    assert [n.value_sum for n in path] == [1.0, 1.0, 1.0]
    assert [n.visit_count for n in path] == [1, 1, 1]


def test_backpropagate_path_flips_exactly_on_player_change() -> None:
    path = [_Node(0), _Node(1), _Node(1), _Node(0)]
    backpropagate_path(path, 1.0)
    # Reference is the leaf's player (0): nodes with player 1 are negated.
    assert [n.value_sum for n in path] == [1.0, -1.0, -1.0, 1.0]


def test_backpropagate_path_falls_back_to_alternation_when_player_unknown() -> None:
    """Callers that never set to_play keep their historical behaviour."""
    path = [_Node(None), _Node(None), _Node(None)]
    backpropagate_path(path, 1.0)
    assert [n.value_sum for n in path] == [1.0, -1.0, 1.0]


def test_backpropagate_path_is_a_noop_on_an_empty_path() -> None:
    backpropagate_path([], 1.0)  # must not raise


def test_backup_edge_summary_counts_edges() -> None:
    same, changed, unknown = backup_edge_summary([_Node(0), _Node(0), _Node(1), _Node(None)])
    assert (same, changed, unknown) == (1, 1, 1)


# --- Partially-populated paths -----------------------------------------------
#
# `backpropagate_path` is exported from `core/mcts` and the `civ6_mcts` shim, so
# a future caller may populate `to_play` on some nodes and not others. The
# original implementation applied the alternation fallback to a RUNNING value,
# so a single unknown node inverted every node above it -- the root of a
# `(known, unknown, known)` path was signed -1 when it should have been +1.
# Signing each node independently from the leaf's perspective keeps the loss of
# information local to the node that cannot be signed.


def test_an_unknown_node_does_not_invert_its_ancestors() -> None:
    path = [_Node(0), _Node(None), _Node(0)]
    backpropagate_path(path, 1.0)
    assert [n.value_sum for n in path] == [1.0, 1.0, 1.0], (
        "an unknown intermediate node inverted the root; the fallback must stay local"
    )


def test_an_unknown_node_does_not_disturb_a_known_opponent_ancestor() -> None:
    path = [_Node(1), _Node(None), _Node(0)]
    backpropagate_path(path, 1.0)
    # Root is the opponent of the leaf player, so it is negated; the unknown
    # node is credited the leaf's own value; the leaf keeps +1.
    assert [n.value_sum for n in path] == [-1.0, 1.0, 1.0]


def test_two_unknown_nodes_do_not_compound() -> None:
    path = [_Node(0), _Node(0), _Node(None), _Node(0)]
    backpropagate_path(path, 1.0)
    assert [n.value_sum for n in path] == [1.0, 1.0, 1.0, 1.0]


def test_all_unknown_still_alternates() -> None:
    """The historical behaviour is preserved exactly when nothing is tracked."""
    path = [_Node(None), _Node(None), _Node(None)]
    backpropagate_path(path, 1.0)
    assert [n.value_sum for n in path] == [1.0, -1.0, 1.0]
