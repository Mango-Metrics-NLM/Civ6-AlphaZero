"""Closed-loop learning oracle: self-play -> train -> does the *prior* know the win?

What this proves
----------------
Every component below is the project's own: ``self_play_game`` (data
collection and value-target assignment), ``ReplayDataset`` + ``train_epoch``
(the loss and optimiser step), ``MicroCivNetwork`` (the real network, shrunk),
and the two production searches (``Civ6MCTS`` and ``GumbelMCTS``). Only the
environment is synthetic. After a few iterations the network's **raw policy
prior, with search switched off**, must put more than 80% of its mass on the
winning action. That is the property a learning loop exists to deliver, and
aggregate metrics cannot see it: a loop with a sign error still produces
plausible losses and win rates.

Why the payoff is *delayed*
---------------------------
``CommitToWinEnv``: three actions, two players alternating per ply. Picking
``WINNING_ACTION`` *commits* the mover to winning, but the game only ends
``FILLER_PLIES`` later (the other player's moves in between change nothing).
With a budget of ``SIMS`` simulations over a 3-way tree the terminal is well
beyond the horizon, so the search cannot see the win directly -- it has to
read it off the **value head**, which has to have learned that "I committed"
is +1 and "the opponent committed" is -1 from the outcomes ``self_play_game``
assigned. That is what makes the negative control bite: with the value-target
sign inverted the value head learns the opposite, the search avoids the
winning action, and the prior never learns it. A one-ply forced win would
pass the control, because the search would read the terminal reward and the
value head would be irrelevant.

What it does not exercise
-------------------------
The replay buffer, the orchestrator/trainer loops around ``train_epoch``, the
MicroCiv encoder, N-player targets, and MuZero. It also does not test play
strength; it tests that a correct signal propagates outcome -> value target ->
search -> policy target -> prior.

Measured when written (CPU, OMP_NUM_THREADS=2): see the accompanying report;
re-run with ``-s`` for the current numbers. Negative control
(``test_inverted_value_sign_does_not_learn``) is a standing test, so it runs
every time rather than once at authoring.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.core.mcts.base import MCTSConfig
from alpha_zero.core.mcts.gumbel import GumbelMCTS, GumbelMCTSConfig
from alpha_zero.core.mcts.puct import Civ6MCTS
from alpha_zero.training_civ6 import (
    ReplayDataset,
    TrainingConfig,
    TrainingExample,
    self_play_game,
    train_epoch,
)

NUM_ACTIONS = 3
WINNING_ACTION = 1  # not 0: a tie-broken argmax must not be able to pass this
BOARD = 4
CHANNELS = 2
VECTOR = 64
FILLER_PLIES = 6
MAX_PLIES = 14
SIMS = 32
ITERATIONS = 4
GAMES_PER_ITERATION = 16
EPOCHS = 4
PRIOR_MASS_REQUIRED = 0.8
SEED = 20261110


class _ActionSpace:
    def __init__(self, n: int) -> None:
        self.n = n


class CommitToWinEnv:
    """Two players alternate; the first to play ``WINNING_ACTION`` wins, later.

    State: who (if anyone) has committed, how many plies remain, whose move it
    is. The observation encodes exactly that -- channel 0 is ``1`` when the
    *mover* has committed, channel 1 when the *opponent* has -- so the value
    head can learn the sign from the mover's perspective, which is the
    perspective ``self_play_game`` assigns targets in.
    """

    num_players = 2

    def __init__(self) -> None:
        self.action_space = _ActionSpace(NUM_ACTIONS)
        self.current_player = 0
        self.turn = 0
        self._plies = 0
        self._committed: int | None = None
        self._plies_since_commit = 0
        self.winner: int | None = None
        self._done = False

    # -- gym-ish surface used by self_play_game -------------------------------
    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.current_player = 0
        self.turn = 0
        self._plies = 0
        self._committed = None
        self._plies_since_commit = 0
        self.winner = None
        self._done = False
        return self.get_observation(), {}

    def clone(self) -> CommitToWinEnv:
        twin = CommitToWinEnv()
        twin.current_player = self.current_player
        twin.turn = self.turn
        twin._plies = self._plies
        twin._committed = self._committed
        twin._plies_since_commit = self._plies_since_commit
        twin.winner = self.winner
        twin._done = self._done
        return twin

    @property
    def done(self) -> bool:
        return self._done

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step after terminal")
        if self._committed is None and int(action) == WINNING_ACTION:
            self._committed = self.current_player
            self._plies_since_commit = 0
        elif self._committed is not None:
            self._plies_since_commit += 1
        self._plies += 1
        self.current_player = 1 - self.current_player
        self.turn = self._plies // 2

        if self._committed is not None and self._plies_since_commit >= FILLER_PLIES:
            self._done = True
            self.winner = self._committed
        elif self._plies >= MAX_PLIES:
            self._done = True
            self.winner = None
        return self.get_observation(), 0.0, self._done, False, self.get_info()

    def get_observation(self) -> dict[str, np.ndarray]:
        spatial = np.zeros((CHANNELS, BOARD, BOARD), dtype=np.float32)
        if self._committed is not None:
            channel = 0 if self._committed == self.current_player else 1
            spatial[channel] = 1.0
        vector = np.zeros(VECTOR, dtype=np.float32)
        vector[0] = self._plies / MAX_PLIES
        vector[1] = float(self.current_player)
        return {"spatial": spatial, "vector": vector, "action_mask": self.get_valid_actions()}

    def get_valid_actions(self) -> np.ndarray:
        return np.ones(NUM_ACTIONS, dtype=np.float32)

    def get_info(self) -> dict[str, Any]:
        return {"winner": self.winner} if self._done else {}


def _network(seed: int) -> MicroCivNetwork:
    torch.manual_seed(seed)
    net = MicroCivNetwork(
        board_size=BOARD,
        action_space_size=NUM_ACTIONS,
        num_res_blocks=1,
        num_filters=8,
        spatial_channels=CHANNELS,
    )
    return net


def _config() -> TrainingConfig:
    return TrainingConfig(
        board_size=BOARD,
        num_simulations=SIMS,
        c_puct=1.5,
        dirichlet_epsilon=0.0,
        temperature=1.0,
        temperature_threshold=2,
        batch_size=32,
        learning_rate=0.01,
        gradient_clip=1.0,
        device="cpu",
        seed=SEED,
    )


def _search(variant: str, network: MicroCivNetwork, seed: int) -> Any:
    if variant == "civ6":
        return Civ6MCTS(
            network,
            MCTSConfig(num_simulations=SIMS, dirichlet_epsilon=0.0, seed=seed),
            device="cpu",
        )
    if variant == "gumbel":
        return GumbelMCTS(
            network,
            GumbelMCTSConfig(num_simulations=SIMS, num_sampled_actions=NUM_ACTIONS, seed=seed),
            device="cpu",
        )
    raise ValueError(variant)


def prior_mass_on_winning_action(network: MicroCivNetwork) -> float:
    """Softmax of the raw policy logits on the initial position; no search."""
    obs = CommitToWinEnv().get_observation()
    network.eval()
    with torch.no_grad():
        logits, _ = network(
            torch.from_numpy(obs["spatial"]).unsqueeze(0),
            torch.from_numpy(obs["vector"]).unsqueeze(0),
            torch.from_numpy(obs["action_mask"]).unsqueeze(0),
        )
    probs = torch.softmax(logits[0], dim=-1).numpy()
    return float(probs[WINNING_ACTION])


def run_learning_loop(variant: str, *, invert_value_sign: bool = False) -> dict[str, float]:
    """Self-play -> train for ``ITERATIONS``; return the prior mass trajectory."""
    config = _config()
    network = _network(SEED)
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    env = CommitToWinEnv()
    trajectory: list[float] = [prior_mass_on_winning_action(network)]
    started = time.perf_counter()
    for iteration in range(ITERATIONS):
        network.eval()
        search = _search(variant, network, SEED + 1000 * iteration)
        examples: list[TrainingExample] = []
        for game in range(GAMES_PER_ITERATION):
            examples.extend(self_play_game(env, search, config, game_seed=SEED + game))
        if invert_value_sign:
            # The negative control: the one place the value target is touched.
            examples = [
                TrainingExample(
                    spatial=e.spatial,
                    vector=e.vector,
                    action_mask=e.action_mask,
                    policy=e.policy,
                    value=-e.value,
                )
                for e in examples
            ]
        generator = torch.Generator().manual_seed(SEED + iteration)
        loader = DataLoader(
            ReplayDataset(examples), batch_size=config.batch_size, shuffle=True, generator=generator
        )
        for _epoch in range(EPOCHS):
            train_epoch(network, optimizer, loader, config)
        trajectory.append(prior_mass_on_winning_action(network))
    return {
        "initial": trajectory[0],
        "final": trajectory[-1],
        "seconds": time.perf_counter() - started,
        "examples_last_iteration": float(len(examples)),
    }


@pytest.fixture(scope="module", params=["civ6", "gumbel"])
def learned(request: pytest.FixtureRequest) -> tuple[str, dict[str, float]]:
    variant = request.param
    result = run_learning_loop(variant)
    print(
        f"\nlearning oracle [{variant}]: prior mass on winning action "
        f"{result['initial']:.3f} -> {result['final']:.3f} after {ITERATIONS} iterations "
        f"({result['seconds']:.1f}s, {int(result['examples_last_iteration'])} examples/iter)"
    )
    return variant, result


def test_fixture_delays_the_payoff_beyond_the_horizon() -> None:
    """Guard the fixture: the win must not be visible as an immediate terminal."""
    env = CommitToWinEnv()
    env.reset()
    _obs, _r, terminated, _t, info = env.step(WINNING_ACTION)
    assert not terminated and info == {}
    for _ in range(FILLER_PLIES - 1):
        _obs, _r, terminated, _t, info = env.step(0)
        assert not terminated
    _obs, _r, terminated, _t, info = env.step(0)
    assert terminated and info["winner"] == 0


def test_fixture_alternates_players() -> None:
    env = CommitToWinEnv()
    env.reset()
    seen = [env.current_player]
    for _ in range(4):
        env.step(0)
        seen.append(env.current_player)
    assert seen == [0, 1, 0, 1, 0]


def test_prior_starts_uninformed(learned: tuple[str, dict[str, float]]) -> None:
    _variant, result = learned
    assert result["initial"] < PRIOR_MASS_REQUIRED, "a fresh network must not already know"


def test_prior_learns_the_winning_action(learned: tuple[str, dict[str, float]]) -> None:
    variant, result = learned
    assert result["final"] > PRIOR_MASS_REQUIRED, (
        f"{variant}: prior mass on the winning action is {result['final']:.3f} after "
        f"{ITERATIONS} iterations; the outcome signal did not reach the policy head"
    )


def test_loop_is_fast_enough_to_gate(learned: tuple[str, dict[str, float]]) -> None:
    _variant, result = learned
    assert result["seconds"] < 60.0


@pytest.mark.parametrize("variant", ["civ6", "gumbel"])
def test_inverted_value_sign_does_not_learn(variant: str) -> None:
    """Standing negative control: the value head is load-bearing.

    With the value targets negated the value head learns that committing is
    bad, the search avoids the winning action, and the prior must fail the
    bar the positive test sets. If this ever passes 0.8 the oracle has stopped
    depending on the value head and no longer proves what it claims.
    """
    result = run_learning_loop(variant, invert_value_sign=True)
    print(
        f"\nlearning oracle NEGATIVE CONTROL [{variant}]: prior mass "
        f"{result['initial']:.3f} -> {result['final']:.3f}"
    )
    assert result["final"] < PRIOR_MASS_REQUIRED
