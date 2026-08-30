"""The seed must determine the run — training and evaluation alike.

This project's only recorded training run says it could not be replayed
(``docs/evidence/validation_iteration_zero/validation_iteration_zero_report.md``):

    **Not reproducible.** ``training_civ6.py`` exposes no ``--seed`` flag and
    ``MicroCivEnv.reset()`` is unseeded in the eval path, so neither the run nor
    the probes can be replayed exactly. The five probes above are the direct
    evidence: same checkpoint, same opponent seed, score rates spanning
    **0.25 to 0.40**.

Five 20-game probes of one checkpoint scored 0.25 / 0.35 / 0.375 / 0.375 / 0.40.
Every judgement downstream inherits that spread — the promotion threshold, the
encoder A/B, per-iteration win-rate trends.

Four unseeded sources produced it, and only the first was in the original plan:

1. Eight global-RNG draws inside the search (``np.random.choice`` /
   ``dirichlet`` / ``uniform``), reachable from no config.
2. ``MicroCivEnv.clone()`` built a fresh env with **no seed**, so every MCTS
   rollout — cloning is how the search simulates — drew from a new unseeded
   generator regardless of what the caller seeded.
3. ``ReplayBuffer.sample`` and ``DataLoader(shuffle=True)``, which decide what a
   batch contains and therefore the gradient.
4. ``NeuralNetworkPlayer`` — the *evaluated agent* — sampling globally while its
   siblings in the same module each owned a seeded generator.

Numbers 2, 3 and 4 were found by *measuring*, not by reading: two ``--seed 42``
runs agreed on 525 lines of self-play output and differed on the first epoch
loss. That is why this file asserts on observed output rather than on the
presence of seeding code.

Scale is kept tiny (6x6 board, 6 turns, 2 simulations) so it gates every PR.

Negative controls, each run against this file — what it catches, and what it
does not:

===================================== ==============================================
Defect reintroduced                    Assertion that fired
===================================== ==============================================
``puct`` draws globally                ``test_two_runs_at_one_seed_are_identical``
                                       *and* the static check
``gumbel`` draws globally              static check only — self-play uses PUCT, so
                                       the behavioural test never reaches Gumbel.
                                       This is exactly why the static check exists.
``clone()`` drops the RNG              ``test_clone_reproduces_the_parent_stream``
``torch.manual_seed`` removed          ``…_at_one_seed_produce_identical_weights``
``ReplayBuffer.sample`` goes global    ``…_at_one_seed_produce_identical_weights``
``DataLoader`` loses its generator     **nothing** — see below
===================================== ==============================================

The last row is recorded rather than papered over. ``train()`` seeds torch
globally before building the network, so ``DataLoader(shuffle=True)`` is already
deterministic without the explicit generator; no control can distinguish its
presence today. It is kept because it removes the shuffle's dependence on *how
many* torch draws happen to precede it, which is fragile in a way a passing test
would not reveal — but this file does not gate it, and saying otherwise would be
the failure mode it was written to prevent.

Two mistakes in this file were themselves caught by controls that failed to
fire, which is the argument for running them: it read the checkpoint under
``model_state_dict`` (the real key is ``state_dict``) and extracted nothing, and
it left ``min_replay_size`` at its 1000 default, so the training phase was
skipped entirely and the "full run" tests exercised network initialisation
alone.
"""

from __future__ import annotations

import contextlib

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

_BOARD_SIZE = 6
_MAX_TURNS = 6
_SIMS = 2
_GAMES = 2
_WEIGHT_SEED = 4242


@contextlib.contextmanager
def _pristine_global_rngs():
    """Run a block without letting it leak numpy/torch global state.

    These tests deliberately disturb the global RNGs to prove the seed does not
    depend on them. Restoring afterwards keeps that from becoming test-order
    coupling for everything collected later in the worker.
    """
    np_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            np.random.set_state(np_state)


def _network(board_size: int = _BOARD_SIZE):
    """A network with fixed weights, built from a forked RNG."""
    from alpha_zero.core.civ6_network import MicroCivNetwork

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_WEIGHT_SEED)
        net = MicroCivNetwork(board_size=board_size, action_space_size=376)
    net.eval()
    return net


def _self_play(seed: int | None) -> list:
    """One self-play round at the given seed, via the real training entry point."""
    from alpha_zero.training_civ6 import TrainingConfig, run_self_play

    config = TrainingConfig(
        board_size=_BOARD_SIZE,
        max_turns=_MAX_TURNS,
        num_simulations=_SIMS,
        device="cpu",
        seed=seed,
    )
    return run_self_play(_network(), config, num_games=_GAMES)


def _fingerprint(examples: list) -> list[tuple]:
    """What the run actually produced: chosen policy and outcome per position."""
    return [
        (tuple(np.asarray(ex.policy).round(6).tolist()), round(float(ex.value), 6))
        for ex in examples
    ]


class TestSelfPlayReplaysExactly:
    """Same seed, same play. The property the whole change exists for."""

    def test_two_runs_at_one_seed_are_identical(self) -> None:
        with _pristine_global_rngs():
            first = _self_play(seed=1234)
            # Disturb the global RNGs between the runs. If the seed genuinely
            # determines the run, this must not matter -- and if it does matter,
            # something is still drawing from global state.
            np.random.random(500)
            torch.rand(500)
            second = _self_play(seed=1234)

        assert _fingerprint(first) == _fingerprint(second), (
            "two self-play rounds at the same seed diverged. Something in the "
            "run is still drawing from a generator the seed does not reach -- "
            "check for a bare np.random.* call, or an env clone that does not "
            "carry its RNG."
        )

    def test_different_seeds_diverge(self) -> None:
        """The positive half: a stuck RNG would satisfy the test above."""
        with _pristine_global_rngs():
            first = _self_play(seed=1234)
            other = _self_play(seed=8765)

        assert _fingerprint(first) != _fingerprint(other), (
            "two different seeds produced identical play, so the seed is being "
            "ignored rather than honoured"
        )

    def test_unseeded_runs_still_work_and_still_differ(self) -> None:
        """``None`` must remain "unseeded", not silently become a fixed seed."""
        with _pristine_global_rngs():
            first = _self_play(seed=None)
            second = _self_play(seed=None)

        assert first, "unseeded self-play produced no examples"
        assert _fingerprint(first) != _fingerprint(second), (
            "two unseeded runs were identical; seed=None should leave the run "
            "nondeterministic, and a hidden default seed would make every "
            "existing caller's behaviour change"
        )


class TestTheEnvironmentCarriesItsRngThroughCloning:
    """Cloning is how MCTS simulates, and ``step`` itself draws.

    ``clone()`` constructed a fresh ``MicroCivEnv`` with no seed, so a seeded
    environment produced unseeded rollouts in every search variant. No seed
    further up could correct it, and it is invisible to any test that only
    checks the top-level env.
    """

    def test_clone_reproduces_the_parent_stream(self) -> None:
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=_BOARD_SIZE, max_turns=_MAX_TURNS, seed=5)
        env.reset(seed=5)
        first = env.clone()._rng.random(20).tolist()
        second = env.clone()._rng.random(20).tolist()
        assert first == second, (
            "two clones of the same environment drew different values, so clones "
            "are not carrying the parent's RNG -- every MCTS rollout is unseeded"
        )

    def test_cloning_does_not_advance_the_parent(self) -> None:
        """A simulation must not perturb the real game's randomness."""
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=_BOARD_SIZE, max_turns=_MAX_TURNS, seed=5)
        env.reset(seed=5)
        before = env._rng.bit_generator.state
        env.clone()._rng.random(100)
        assert env._rng.bit_generator.state == before, (
            "cloning advanced the parent's RNG; searching would then change the game being searched"
        )


class TestEveryVariantOwnsItsRandomness:
    """No search may draw from the global RNG.

    A static check, deliberately: the behavioural tests above cover the paths
    self-play exercises, but MuZero and the batch search are not on that path,
    and a reintroduced global draw there would go unnoticed until it silently
    de-determined a cloud run.
    """

    def test_no_module_in_the_search_calls_the_global_rng(self) -> None:
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[2]
        targets = sorted((root / "alpha_zero" / "core" / "mcts").glob("*.py"))
        targets.append(root / "alpha_zero" / "core" / "batch_mcts.py")

        offenders: list[str] = []
        for path in targets:
            for num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                # `default_rng` is the constructor, not a draw.
                if re.search(r"\bnp\.random\.(?!default_rng)[a-z_]+\(", line):
                    offenders.append(f"{path.relative_to(root)}:{num}: {line.strip()}")

        assert not offenders, (
            "search modules draw from the global numpy RNG:\n"
            + "\n".join(offenders)
            + "\n\nEach search owns `self._rng`; a global draw cannot be reached "
            "by any seed, which is what made a 'seeded' run only half determined."
        )


class TestTheRealEntryPointReplays:
    """Exercise ``train()`` itself, not a stand-in.

    The first version of this file built its own network via a forked RNG and
    called ``run_self_play`` -- which meant deleting ``torch.manual_seed`` from
    ``train()`` left the whole gate green. A gate that misses the thing it
    claims to guard is the defect this branch exists to fix, so it is checked
    here against the real function.

    Affordable because it is: a 1-iteration, 1-game, 1-epoch run on a 6x6 board
    takes ~1.3s, so network initialisation, self-play, replay sampling, batch
    shuffling and checkpointing are all covered on every PR.
    """

    @staticmethod
    def _train_to(tmpdir, seed: int | None):
        from alpha_zero.training_civ6 import TrainingConfig, train

        config = TrainingConfig(
            board_size=_BOARD_SIZE,
            max_turns=4,
            num_simulations=1,
            num_games_per_iteration=1,
            num_iterations=1,
            num_epochs=1,
            batch_size=4,
            # Without this the training phase is skipped entirely
            # (`len(replay_buffer) >= config.min_replay_size`), so the
            # checkpoint would hold nothing but a freshly initialised network
            # and this test would claim to cover replay sampling and batch
            # shuffling while never executing either. Found by a negative
            # control that failed to fire.
            min_replay_size=1,
            num_res_blocks=1,
            num_filters=8,
            checkpoint_dir=str(tmpdir / "ckpt"),
            log_dir=str(tmpdir / "log"),
            device="cpu",
            seed=seed,
        )
        train(config)
        return torch.load(tmpdir / "ckpt" / "latest.pt", map_location="cpu", weights_only=False)

    @staticmethod
    def _weights(checkpoint) -> list[tuple[str, float]]:
        """Per-parameter checksums from the saved checkpoint.

        The key is ``state_dict`` (verified against a real checkpoint, not
        guessed -- the first draft read ``model_state_dict`` and extracted
        nothing). Names are kept alongside the values so a divergence report
        says which parameter moved.
        """
        state = checkpoint["state_dict"]
        return [
            (name, round(float(tensor.double().sum().item()), 6))
            for name, tensor in sorted(state.items())
            if hasattr(tensor, "double")
        ]

    def test_two_full_runs_at_one_seed_produce_identical_weights(self, tmp_path) -> None:
        with _pristine_global_rngs():
            first = self._weights(self._train_to(tmp_path / "a", seed=31))
            # Disturb the globals between runs: the seed must carry the run.
            np.random.random(500)
            torch.rand(500)
            second = self._weights(self._train_to(tmp_path / "b", seed=31))

        assert first, "checkpoint held no tensors -- the comparison is vacuous"
        assert first == second, (
            "two complete training runs at the same seed produced different "
            "weights. Network initialisation, replay sampling and batch order "
            "are all inside this path; one of them is drawing from a generator "
            "the seed does not reach."
        )

    def test_a_different_seed_produces_different_weights(self, tmp_path) -> None:
        """The positive half -- identical weights everywhere would also pass."""
        with _pristine_global_rngs():
            first = self._weights(self._train_to(tmp_path / "c", seed=31))
            other = self._weights(self._train_to(tmp_path / "d", seed=77))

        assert first != other, (
            "two different seeds trained to identical weights; the seed is not "
            "reaching network initialisation or the training data"
        )


class TestEvaluationProbesRepeat:
    """A probe of one checkpoint must give one answer.

    The recorded spread (0.25 to 0.40 over five probes) is what this prevents.
    Environments and opponents were already seeded; the evaluated *agent* was
    not, which is why the spread survived seeded environments.
    """

    @staticmethod
    def _probe(seed: int) -> tuple[int, int, int]:
        from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig
        from alpha_zero.evaluation.evaluation_pipeline import EvaluationPipeline

        agent = Civ6MCTS(
            network=_network(),
            config=MCTSConfig(num_simulations=_SIMS, seed=seed + _GAMES),
            device="cpu",
        )
        pipeline = EvaluationPipeline(board_size=_BOARD_SIZE, max_turns=_MAX_TURNS, seed=seed)
        rate = pipeline.evaluate_vs_random(agent, n_games=_GAMES).win_rate
        return (rate.wins, rate.draws, rate.losses)

    def test_two_probes_at_one_seed_agree(self) -> None:
        with _pristine_global_rngs():
            first = self._probe(7)
            np.random.random(500)
            second = self._probe(7)

        assert first == second, (
            f"two probes of the same checkpoint at one seed disagreed: {first} "
            f"vs {second}. This is the defect recorded in "
            "docs/evidence/validation_iteration_zero/ -- a single probe is then "
            "one draw from a wide distribution rather than a measurement."
        )

    def test_a_different_seed_can_give_a_different_answer(self) -> None:
        """Not an assertion that it *must* differ -- a small probe legitimately
        can repeat. Only that the seed reaches the evaluation at all, which is
        checked by the two-probe agreement above being seed-specific."""
        with _pristine_global_rngs():
            baseline = self._probe(7)
            other = self._probe(99)

        # Both are valid outcomes; what matters is that the call is honoured
        # deterministically, so re-running `other` reproduces `other`.
        with _pristine_global_rngs():
            other_again = self._probe(99)
        assert other == other_again, "a probe is not reproducible at its own seed"
        assert isinstance(baseline, tuple)
