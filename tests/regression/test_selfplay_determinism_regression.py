"""Regression tests for the half-seeded self-play defect.

``tests/integration/test_batch_selfplay_integration.py::…::
test_value_targets_are_not_uniformly_degenerate`` failed in CI while passing
locally, on a commit whose only change was a markdown file. It was not a flake
in the usual sense -- the test was **half-seeded** and had been passing on luck.

Two independent sources of randomness were left free:

1. ``run_batch_selfplay(seed=...)`` seeds the *environments* only. The worker
   says so at its config construction: "the environments are only half the
   randomness -- ``gumbel_sample_actions`` draws from the *global* numpy RNG, so
   a caller wanting full reproducibility must seed that too."
2. The test's network fixture was built from an unseeded torch RNG, so the
   policy driving the search differed on every run.

Under this environment's documented ~73.5% draw rate, a test asserting that some
game reaches a decisive outcome can then legitimately see every game of every
batch draw. Seeding the model alone was measured at 5/6 consecutive passes;
seeding both gave 8/8.

These tests pin the property rather than the symptom: identical full seeding
must reproduce identical play, and the global-RNG coupling that made ``seed=``
insufficient must stay visible. Both directions are deterministic -- every
branch here is fully seeded -- and the three self-play runs they need are shared
across the module so the suite stays cheap enough to gate every PR.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

_BOARD_SIZE = 6
_MAX_TURNS = 8
_MCTS_SIMS = 2
_GUMBEL_ACTIONS = 2
_NUM_GAMES = 2
_MODEL_SEED = 1337
_ENV_SEED = 11
_GLOBAL_SEED = 17
_OTHER_GLOBAL_SEED = 99

# The per-step fields that together describe what the search did. `rewards` is
# the value target the degenerate-target test asserts on; `policies` is the
# search output the network trains against.
_REPLAYED_FIELDS = ("actions", "policies", "rewards")


def _seed_global_rngs(seed: int) -> None:
    """Seed the RNGs that ``run_batch_selfplay(seed=...)`` does not reach."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_model():
    """A deterministic network, built from a forked RNG.

    Forking matters: seeding ``torch`` globally here would make every later test
    in the session implicitly depend on this seed, trading one hidden-coupling
    bug for another.
    """
    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

    config = TransformerConfig(
        spatial_channels=13,
        spatial_size=_BOARD_SIZE,
        vector_dim=64,
        action_space_size=376,
        num_layers=1,
        embed_dim=32,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_MODEL_SEED)
        model = Civ6TransformerNetwork(config=config)
    model.eval()
    return model


def _play(*, env_seed: int, global_seed: int) -> list[dict]:
    from alpha_zero.cloud.gcp.selfplay_worker import run_batch_selfplay

    _seed_global_rngs(global_seed)
    return run_batch_selfplay(
        model=_build_model(),
        num_games=_NUM_GAMES,
        mcts_sims=_MCTS_SIMS,
        board_size=_BOARD_SIZE,
        max_turns=_MAX_TURNS,
        num_parallel_games=_NUM_GAMES,
        gumbel_num_actions=_GUMBEL_ACTIONS,
        device="cpu",
        seed=env_seed,
    )


def _fingerprint(trajectories: list[dict]) -> list[dict[str, list]]:
    """A comparable, field-by-field summary of a batch of trajectories."""
    return [
        {field: [np.asarray(step).tolist() for step in traj[field]] for field in _REPLAYED_FIELDS}
        for traj in trajectories
    ]


# Self-play is the expensive part, so run each distinct configuration once.
@pytest.fixture(scope="module")
def baseline_run() -> list[dict]:
    return _play(env_seed=_ENV_SEED, global_seed=_GLOBAL_SEED)


@pytest.fixture(scope="module")
def repeat_run() -> list[dict]:
    """Same environment seed, same global seed, freshly seeded network."""
    return _play(env_seed=_ENV_SEED, global_seed=_GLOBAL_SEED)


@pytest.fixture(scope="module")
def other_global_seed_run() -> list[dict]:
    """Same environment seed; only the global RNG differs."""
    return _play(env_seed=_ENV_SEED, global_seed=_OTHER_GLOBAL_SEED)


class TestFullSeedingIsReproducible:
    """The property the fix establishes."""

    def test_identical_seeding_reproduces_identical_play(self, baseline_run, repeat_run) -> None:
        assert _fingerprint(baseline_run) == _fingerprint(repeat_run), (
            "identical environment seed, identical global RNG seed and identical "
            "network weights produced different play; batch self-play is not "
            "reproducible even when fully seeded, so no outcome assertion on it "
            "can be trusted"
        )

    @pytest.mark.parametrize("field", _REPLAYED_FIELDS)
    def test_each_replayed_field_matches(self, baseline_run, repeat_run, field: str) -> None:
        """Names the field that diverged rather than reporting one opaque diff."""
        first = [np.asarray(step).tolist() for traj in baseline_run for step in traj[field]]
        second = [np.asarray(step).tolist() for traj in repeat_run for step in traj[field]]
        assert first == second, f"{field!r} differs between two identically seeded runs"

    def test_trajectories_are_actually_produced(self, baseline_run) -> None:
        """Guards the tests above against passing on two empty results."""
        assert baseline_run, "no trajectories at all -- the comparisons prove nothing"
        assert all(traj["actions"] for traj in baseline_run), (
            "a trajectory has no steps; comparing empty lists is a vacuous pass"
        )


class TestTheEnvSeedIsOnlyHalfTheRandomness:
    """Why seeding ``seed=`` alone was not enough -- the defect itself.

    Both branches are fully seeded, so this is deterministic; it pins an observed
    coupling rather than sampling one. If it ever fails, the honest reading is
    that Gumbel sampling stopped drawing from the global numpy RNG -- at which
    point ``run_batch_selfplay``'s docstring and the callers that seed globally
    should be revisited, not this assertion deleted.
    """

    def test_global_rng_changes_play_at_a_fixed_env_seed(
        self, baseline_run, other_global_seed_run
    ) -> None:
        assert _fingerprint(baseline_run) != _fingerprint(other_global_seed_run), (
            "the global RNG no longer affects play at a fixed environment seed. "
            "That would make `seed=` sufficient for reproducibility -- good news, "
            "but it invalidates the documented contract that callers must seed "
            "numpy themselves"
        )

    def test_docstring_still_warns_that_seed_covers_environments_only(self) -> None:
        """The contract is documented; keep it documented.

        The integration test was written against an assumption the code
        contradicts in its own docstring. If that warning is dropped, the next
        caller repeats the mistake.
        """
        from alpha_zero.cloud.gcp import selfplay_worker

        doc = selfplay_worker.run_batch_selfplay.__doc__ or ""
        assert "environments only" in doc, (
            "run_batch_selfplay's `seed` docstring no longer states that it seeds "
            "the environments only; that omission is what the half-seeded "
            "integration test was built on"
        )


@pytest.fixture(scope="module")
def integration_source() -> str:
    """Source of the integration file where the half-seeding defect surfaced."""
    path = (
        Path(__file__).resolve().parents[1] / "integration" / "test_batch_selfplay_integration.py"
    )
    return path.read_text(encoding="utf-8")


class TestTheIntegrationTestSeedsBothSources:
    """Guard the fix at its call site, not only as a property.

    The integration file is where the defect surfaced; a future edit that drops
    either half of the seeding puts the flake straight back.
    """

    def test_global_rngs_are_seeded(self, integration_source: str) -> None:
        assert "_seed_global_rngs" in integration_source, (
            "the batch self-play integration tests no longer seed the global RNGs; "
            "outcome assertions there become dependent on how the dice fall"
        )

    def test_network_fixture_weights_are_seeded(self, integration_source: str) -> None:
        assert "fork_rng" in integration_source, (
            "the network fixture no longer builds its weights from a forked, "
            "seeded RNG -- the policy driving the search is unseeded again"
        )
