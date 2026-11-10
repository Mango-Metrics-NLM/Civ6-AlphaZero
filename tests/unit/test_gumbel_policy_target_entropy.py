"""How much of the search does the Gumbel policy target actually carry?

The 2026-09-04 fix replaced a one-hot target (measured: 1 non-zero entry of
376) with Danihelka's completed-Q softmax. This file asserts only the
*contract* of that target on real MicroCiv root positions -- more than one
action carries mass, the entropy is finite, the vector is a distribution over
legal actions -- and **prints** the measured normalised entropy and maximum
mass so the numbers are visible in the test log without being pinned. Pinning
them would turn a hyper-parameter (``c_scale``, see the deviations recorded in
``gumbel_ops.build_completed_q_policy``) into a regression, and the point of
measuring is to be able to change it on evidence.

Measured when this file was written (seed 20261110, 10 positions, 32 sims,
k=16, ``c_scale`` 1.0): see the report that accompanied it; re-run with ``-s``
to print the current values.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.core.mcts.gumbel import GumbelMCTS, GumbelMCTSConfig
from alpha_zero.envs.microciv import MicroCivEnv

_BOARD = 8
_TURNS = 30
_SIMS = 32
_K = 16
_POSITIONS = 10
_SEED = 20261110


def _normalised_entropy(policy: np.ndarray) -> float:
    support = policy[policy > 0]
    if support.size <= 1:
        return 0.0
    return float(-(support * np.log(support)).sum() / np.log(support.size))


@pytest.fixture(scope="module")
def measurements() -> list[dict[str, float]]:
    """Search ten seeded root positions along one random-play game."""
    import torch

    torch.manual_seed(_SEED)
    env = MicroCivEnv(board_size=_BOARD, max_turns=_TURNS, seed=_SEED)
    network = MicroCivNetwork(
        board_size=_BOARD,
        action_space_size=int(env.action_space.n),
        num_res_blocks=2,
        num_filters=16,
    )
    network.eval()
    search = GumbelMCTS(
        network,
        GumbelMCTSConfig(num_simulations=_SIMS, num_sampled_actions=_K, seed=_SEED),
        device="cpu",
    )
    rng = np.random.default_rng(_SEED)
    env.reset(seed=_SEED)
    rows: list[dict[str, float]] = []
    while len(rows) < _POSITIONS:
        mask = env.get_valid_actions()
        policy, _value = search.search(env)
        policy = np.asarray(policy, dtype=np.float64)
        rows.append(
            {
                "legal": float(mask.sum()),
                "nonzero": float((policy > 0).sum()),
                "max_mass": float(policy.max()),
                "entropy": _normalised_entropy(policy),
                "sum": float(policy.sum()),
                "mass_on_illegal": float(policy[mask <= 0].sum()),
            }
        )
        # Advance with a random legal move so the ten roots are distinct positions.
        legal = np.flatnonzero(mask)
        _obs, _r, terminated, truncated, _info = env.step(int(rng.choice(legal)))
        if terminated or truncated:
            env.reset(seed=_SEED + len(rows))
    return rows


def test_target_is_a_distribution_over_legal_actions(
    measurements: list[dict[str, float]],
) -> None:
    for row in measurements:
        assert row["sum"] == pytest.approx(1.0, abs=1e-5)
        assert row["mass_on_illegal"] == pytest.approx(0.0, abs=1e-7)


def test_target_is_not_a_delta(measurements: list[dict[str, float]]) -> None:
    """The contract the 2026-09-04 fix established: more than one action carries mass.

    Only where more than one action is legal: a position whose sole legal move
    is END_TURN correctly yields a delta, and the first run of this file met
    exactly one such position in ten.
    """
    informative = [row for row in measurements if row["legal"] > 1]
    assert len(informative) >= _POSITIONS // 2, "too few non-trivial positions to measure"
    nonzero = [row["nonzero"] for row in informative]
    assert all(n > 1 for n in nonzero), nonzero
    assert all(np.isfinite(row["entropy"]) for row in measurements)


def test_report_measured_entropy(measurements: list[dict[str, float]]) -> None:
    """Prints the measurement; asserts only that it was taken on enough positions."""
    informative = [row for row in measurements if row["legal"] > 1]
    entropy = np.array([row["entropy"] for row in informative])
    max_mass = np.array([row["max_mass"] for row in informative])
    nonzero = np.array([row["nonzero"] for row in informative])
    legal = np.array([row["legal"] for row in informative])
    print(
        "\nGumbel policy target @ "
        f"{_SIMS} sims, k={_K}, {len(informative)} positions with >1 legal action "
        f"(of {len(measurements)}): "
        f"normalised entropy mean={entropy.mean():.3f} min={entropy.min():.3f} "
        f"max={entropy.max():.3f}; max mass mean={max_mass.mean():.3f} "
        f"max={max_mass.max():.3f}; non-zero mean={nonzero.mean():.1f} of "
        f"{legal.mean():.1f} legal"
    )
    assert len(measurements) == _POSITIONS
