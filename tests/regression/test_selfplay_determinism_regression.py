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

import contextlib
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

# The per-step fields that together describe what the search did. `rewards` is
# the value target the degenerate-target test asserts on; `policies` is the
# search output the network trains against.
_REPLAYED_FIELDS = ("actions", "policies", "rewards")


@contextlib.contextmanager
def _seed_global_rngs(seed: int):
    """Seed the RNGs ``run_batch_selfplay(seed=...)`` does not reach, then restore.

    Restoring is the half that is easy to forget. Seeding the process-global
    numpy and torch RNGs and leaving them there would hand every subsequently
    collected test the state this file's self-play happened to end on -- hidden
    test-order coupling, and exactly what :func:`_build_model` uses ``fork_rng``
    to avoid two functions above. A file arguing for RNG hygiene has to keep it.
    """
    np_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            np.random.set_state(np_state)


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

    with _seed_global_rngs(global_seed):
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


def _play_unseeded_globals(*, env_seed: int) -> list[dict]:
    """Self-play with ONLY ``seed=`` set -- no global seeding at all.

    The counterpart to :func:`_play`: it proves the parameter carries the run on
    its own, which is the property that replaced the half-seeded contract.
    """
    from alpha_zero.cloud.gcp.selfplay_worker import run_batch_selfplay

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


class TestTheSeedNowCoversTheWholeRun:
    """The contract that replaced the half-seeded one.

    This class used to assert the *defect*: that ``run_batch_selfplay(seed=...)``
    covered the environments only, and that the global numpy RNG still moved the
    play. Its own comment said what to do if that ever stopped being true --
    "revisit ``run_batch_selfplay``'s docstring and the callers that seed
    globally, not this assertion". That is what happened: each search now owns a
    generator seeded from the same value, so the coupling is gone by design.

    Inverted rather than deleted, because the new property is the stronger one
    and is worth pinning: ``seed=`` alone is now sufficient.
    """

    def test_seed_alone_reproduces_play_without_seeding_numpy(self) -> None:
        """No global seeding anywhere -- the parameter must carry the whole run.

        Deliberately does *not* use ``_seed_global_rngs``: if this passes while
        the global RNGs are left wherever the previous test left them, the seed
        genuinely determines the run on its own.
        """
        first = _play_unseeded_globals(env_seed=_ENV_SEED)
        np.random.random(1000)  # disturb the global state between the two runs
        torch.rand(1000)
        second = _play_unseeded_globals(env_seed=_ENV_SEED)
        assert _fingerprint(first) == _fingerprint(second), (
            "run_batch_selfplay(seed=...) no longer determines the run by itself. "
            "It used to cover the environments only, which is the defect this "
            "file was written for; if that has returned, the search is drawing "
            "from a generator the seed does not reach."
        )

    def test_different_seeds_still_differ(self) -> None:
        """The positive half -- a stuck RNG would satisfy the test above."""
        first = _play_unseeded_globals(env_seed=_ENV_SEED)
        other = _play_unseeded_globals(env_seed=_ENV_SEED + 500)
        assert _fingerprint(first) != _fingerprint(other), (
            "two different seeds produced identical play; the seed is being "
            "ignored rather than honoured"
        )

    def test_docstring_no_longer_tells_callers_to_seed_numpy(self) -> None:
        """The old docstring instructed callers to seed numpy themselves.

        Leaving that text in place after the fix would keep sending readers to a
        workaround that is now both unnecessary and misleading -- global seeding
        no longer reaches the search at all.
        """
        from alpha_zero.cloud.gcp import selfplay_worker

        doc = selfplay_worker.run_batch_selfplay.__doc__ or ""
        assert "determining the whole run" in doc, (
            "run_batch_selfplay's `seed` docstring should state that it determines the whole run"
        )
        assert "must seed that too" not in doc, (
            "the docstring still tells callers to seed the global numpy RNG; that "
            "advice no longer works, because the search owns its own generator"
        )


class TestSeedingDoesNotLeakIntoOtherTests:
    """Seeding is scoped; the process-global RNGs come back unchanged.

    Review caught the first version of this file leaving numpy and torch seeded
    at whatever state self-play ended on, so every test collected afterwards in
    the same worker inherited it. That is the coupling ``_build_model`` uses
    ``fork_rng`` to avoid -- argued for in a docstring and then broken by the
    helper beside it.
    """

    def test_numpy_state_is_restored(self) -> None:
        np.random.seed(4242)
        before = np.random.get_state()
        with _seed_global_rngs(7):
            np.random.random(10)
        after = np.random.get_state()
        assert before[0] == after[0]
        assert np.array_equal(before[1], after[1]), (
            "_seed_global_rngs left numpy's global RNG reseeded; later tests in "
            "this worker would inherit state from this file"
        )
        assert before[2:] == after[2:]

    def test_torch_state_is_restored(self) -> None:
        torch.manual_seed(4242)
        before = torch.get_rng_state()
        with _seed_global_rngs(7):
            torch.rand(10)
        assert torch.equal(before, torch.get_rng_state()), (
            "_seed_global_rngs left torch's global RNG reseeded"
        )

    def test_the_seed_still_takes_effect_inside_the_block(self) -> None:
        """The positive half: a helper that restored but never seeded would
        also pass the two tests above."""
        with _seed_global_rngs(7):
            first = np.random.random(5).tolist()
        with _seed_global_rngs(7):
            second = np.random.random(5).tolist()
        assert first == second, "_seed_global_rngs is not actually seeding"


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
